//! gpu.rs
//! GPU‑accelerated KV subset selection and attention mask building.
//!
//! Uses NVIDIA CUDA through the `cust` crate.
//! Falls back to CPU if CUDA is unavailable.

use kv_web_core::{WebNodeId, KvWeb};
use kv_web_runtime::KvWebRuntime;

// === REQUIRED FOR cust 0.3.x ===
use cust::prelude::*;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use cust::launch;

use std::error::Error;

/// GPU context wrapper.
pub struct GpuContext {
    pub device: Device,
    pub ctx: Context,
}

impl GpuContext {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        cust::init(cust::CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let ctx = Context::new(device)?;
        Ok(Self { device, ctx })
    }
}

/// Optimization config for GPU mask building.
#[derive(Debug, Clone)]
pub struct GpuOptimizationConfig {
    pub min_region_batch: usize,
    pub max_region_batch: usize,
    pub min_gpu_threshold: usize,
    pub max_gpu_threshold: usize,
    pub min_block_size: u32,
    pub max_block_size: u32,
}

/// GPU optimization state.
#[derive(Debug, Clone)]
pub struct GpuOptimizationState {
    pub region_batch: usize,
    pub gpu_threshold: usize,
    pub block_size: u32,
}

impl Default for GpuOptimizationState {
    fn default() -> Self {
        Self {
            region_batch: 256,
            gpu_threshold: 512,
            block_size: 1,
        }
    }
}

/// GPU load metrics for dynamic balancing.
#[derive(Debug, Clone)]
pub struct GpuLoad {
    pub sm_count: u32,
    pub warp_size: u32,
    pub max_threads_per_sm: u32,
    pub current_load: f32,
}

/// Read GPU load using device attributes (placeholder current_load).
pub fn read_gpu_load(device: &Device) -> GpuLoad {
    let sm_count = device
        .get_attribute(DeviceAttribute::MultiprocessorCount)
        .unwrap_or(0) as u32;
    let warp_size = device
        .get_attribute(DeviceAttribute::WarpSize)
        .unwrap_or(32) as u32;
    let max_threads_per_sm = device
        .get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor)
        .unwrap_or(0) as u32;

    // Placeholder: real load measurement would use NVML or CUDA perf counters.
    let current_load = 0.35;

    GpuLoad {
        sm_count,
        warp_size,
        max_threads_per_sm,
        current_load,
    }
}

/// Partition region tokens into balanced chunks based on GPU load.
pub fn partition_region_for_gpu(
    region: &[u32],
    load: &GpuLoad,
    state: &GpuOptimizationState,
) -> Vec<Vec<u32>> {
    let base_batch = state.region_batch;

    let adjusted_batch = if load.current_load > 0.75 {
        base_batch / 4
    } else if load.current_load > 0.50 {
        base_batch / 2
    } else {
        base_batch
    };

    let mut chunks = Vec::new();
    let mut i = 0;

    while i < region.len() {
        let end = (i + adjusted_batch).min(region.len());
        chunks.push(region[i..end].to_vec());
        i = end;
    }

    chunks
}

/// Multi‑stream GPU execution with load balancing.
pub fn build_attention_mask_gpu_balanced(
    region_chunks: &[Vec<u32>],
    kv_len: usize,
    module: &Module,
) -> Vec<f32> {
    let func = module.get_function("build_mask").unwrap();

    let mut streams = Vec::new();
    for _ in region_chunks {
        streams.push(Stream::new(StreamFlags::NON_BLOCKING, None).unwrap());
    }

    let mask_buf = DeviceBuffer::<f32>::zeroed(kv_len).unwrap();

    for (i, chunk) in region_chunks.iter().enumerate() {
        let region_buf = DeviceBuffer::from_slice(chunk).unwrap();
        let region_len = chunk.len() as u32;

        let stream_ref = &streams[i];

        unsafe {
            launch!(
                func<<<region_len, 1, 0, stream_ref>>>(
                    region_buf.as_device_ptr(),
                    region_len,
                    mask_buf.as_device_ptr(),
                    kv_len as u32
                )
            )
            .unwrap();
        }
    }

    for s in streams {
        s.synchronize().unwrap();
    }

    let mut mask = vec![0.0f32; kv_len];
    mask_buf.copy_to(&mut mask).unwrap();

    mask
}

/// GPU‑accelerated attention mask builder.
/// CPU fallback if GPU unavailable.
///
/// Signature kept compatible with existing call sites.
pub fn build_attention_mask_gpu(
    web: &KvWeb,
    root: WebNodeId,
    depth: usize,
    kv_len: usize,
    gpu: Option<&GpuContext>,
) -> Vec<f32> {
    let region = web.tokens_in_region(root, depth);

    // CPU fallback
    if gpu.is_none() {
        let mut mask = vec![0.0; kv_len];
        for t in region {
            if t.0 < kv_len {
                mask[t.0] = 1.0;
            }
        }
        return mask;
    }

    let gpu_ctx = gpu.unwrap();

    // Local max‑tier optimization state + config
    let mut state = GpuOptimizationState::default();
    let cfg = GpuOptimizationConfig {
        min_region_batch: 64,
        max_region_batch: 4096,
        min_gpu_threshold: 128,
        max_gpu_threshold: 16384,
        min_block_size: 1,
        max_block_size: 1024,
    };

    // Runtime-only tuning
    optimize_gpu(web, root, depth, kv_len, &mut state, &cfg);

    let region_vec: Vec<u32> = region.iter().map(|t| t.0 as u32).collect();

    // CUDA kernel source (simple mask builder)
    let ptx = r#"
    .version 7.0
    .target sm_70
    .address_size 64

    .visible .entry build_mask(
        .param .u64 region_ptr,
        .param .u32 region_len,
        .param .u64 mask_ptr,
        .param .u32 kv_len
    )
    {
        .reg .u32 tid;
        .reg .u32 idx;
        .reg .u64 rptr;
        .reg .u64 mptr;

        ld.param.u64 rptr, [region_ptr];
        ld.param.u64 mptr, [mask_ptr];
        ld.param.u32 region_len, [region_len];
        ld.param.u32 kv_len, [kv_len];

        mov.u32 tid, %tid.x;

        setp.ge.u32 p_exit, tid, region_len;
        @p_exit bra DONE;

        mul.lo.u64 idx, tid, 4;
        ld.global.u32 idx, [rptr + idx];

        setp.ge.u32 p_skip, idx, kv_len;
        @p_skip bra DONE;

        mul.lo.u64 idx, idx, 4;
        st.global.f32 [mptr + idx], 1.0;

    DONE:
        ret;
    }
    "#;

    let module = Module::from_ptx(ptx, &[]).unwrap();

    // GPU load balancing
    let load = read_gpu_load(&gpu_ctx.device);
    let region_chunks = partition_region_for_gpu(&region_vec, &load, &state);

    build_attention_mask_gpu_balanced(&region_chunks, kv_len, &module)
}

// ============================================================================
// ⭐ MAX‑TIER GPU OPTIMIZATION LOOP (runtime-only tuning)
// ============================================================================

/// Max-tier optimization loop for GPU mask building.
/// Tunes GPU/CPU crossover, batching, and block size.
pub fn optimize_gpu(
    web: &KvWeb,
    root: WebNodeId,
    depth: usize,
    _kv_len: usize,
    state: &mut GpuOptimizationState,
    cfg: &GpuOptimizationConfig,
) {
    let region = web.tokens_in_region(root, depth);
    let region_size = region.len();

    // 1) GPU threshold tuning
    if region_size < cfg.min_gpu_threshold {
        state.gpu_threshold =
            ((state.gpu_threshold as f32 * 0.9) as usize).max(cfg.min_gpu_threshold);
    } else if region_size > cfg.max_gpu_threshold {
        state.gpu_threshold =
            ((state.gpu_threshold as f32 * 1.1) as usize).min(cfg.max_gpu_threshold);
    }

    // 2) Region batching tuning
    if region_size > state.region_batch {
        state.region_batch = (state.region_batch * 2).min(cfg.max_region_batch);
    } else {
        state.region_batch = (state.region_batch / 2).max(cfg.min_region_batch);
    }

    // 3) Block size tuning
    if region_size > 1024 {
        state.block_size = (state.block_size * 2).min(cfg.max_block_size);
    } else {
        state.block_size = (state.block_size / 2).max(cfg.min_block_size);
    }

    // No compression here — GPU tuning is runtime-only.
}
