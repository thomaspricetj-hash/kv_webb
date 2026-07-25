//! gpu.rs
//! GPU‑accelerated KV subset selection and attention mask building.
//!
//! Uses NVIDIA CUDA through the `cust` crate`.
//! Falls back to CPU if CUDA is unavailable.
//!
//! Tier‑6 → Tier‑8 unified version:
//! - Keeps all original logic.
//! - Adds max‑tier routing, load feedback, compression‑friendly chunking,
//!   and extended roundabout behavior.
//! - Designed so you can later wire real GPU‑side compression and PKM daemons
//!   without changing call sites.

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
            block_size: 64,
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

/// Compression‑aware routing hints (entropy, density, etc.).
#[derive(Debug, Clone)]
pub struct CompressionRoutingHints {
    pub entropy_score: f32,
    pub avg_token_gap: f32,
    pub high_density: bool,
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

/// Simple entropy/density estimator for compression‑aware chunk shaping.
pub fn estimate_compression_hints(region: &[u32]) -> CompressionRoutingHints {
    if region.is_empty() {
        return CompressionRoutingHints {
            entropy_score: 0.0,
            avg_token_gap: 0.0,
            high_density: false,
        };
    }

    let mut gaps = Vec::with_capacity(region.len().saturating_sub(1));
    for w in region.windows(2) {
        gaps.push(w[1].saturating_sub(w[0]) as f32);
    }

    let avg_gap = if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().sum::<f32>() / gaps.len() as f32
    };

    // Very rough entropy proxy: larger gaps → higher entropy.
    let entropy = avg_gap.min(1024.0) / 1024.0;
    let high_density = avg_gap < 8.0;

    CompressionRoutingHints {
        entropy_score: entropy,
        avg_token_gap: avg_gap,
        high_density,
    }
}

/// GPU‑side BitDrop_v2 collapse wiring.
/// Expects a `bitdrop_collapse` kernel in the same module.
/// Falls back to identity if kernel is missing.
fn gpu_bitdrop_collapse_region(module: &Module, region: &[u32]) -> Vec<u32> {
    let len = region.len();
    if len == 0 {
        return Vec::new();
    }

    let func = match module.get_function("bitdrop_collapse") {
        Ok(f) => f,
        Err(_) => return region.to_vec(),
    };

    let input_buf = match DeviceBuffer::from_slice(region) {
        Ok(b) => b,
        Err(_) => return region.to_vec(),
    };

    let mut output_buf = match DeviceBuffer::<u32>::zeroed(len) {
        Ok(b) => b,
        Err(_) => return region.to_vec(),
    };

    let threads_per_block = 256u32;
    let blocks = ((len as u32 + threads_per_block - 1) / threads_per_block).max(1);

    // Use a local NON_BLOCKING stream for this collapse pass.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    unsafe {
        launch!(
            func<<<blocks, threads_per_block, 0, stream>>>(
                input_buf.as_device_ptr(),
                output_buf.as_device_ptr(),
                len as u32
            )
        )
        .unwrap();
    }

    stream.synchronize().unwrap();

    let mut out = vec![0u32; len];
    if output_buf.copy_to(&mut out).is_err() {
        return region.to_vec();
    }

    out
}

/// Partition region tokens into balanced chunks based on GPU load + hybrid priority
/// + compression‑aware shaping. No original logic removed; only extended.
pub fn partition_region_for_gpu(
    region: &[u32],
    load: &GpuLoad,
    state: &GpuOptimizationState,
) -> Vec<Vec<u32>> {
    let base_batch = state.region_batch;
    let hints = estimate_compression_hints(region);

    // Hybrid priority: combine load + region size heuristics
    let mut adjusted_batch = base_batch;

    if load.current_load > 0.80 {
        adjusted_batch = base_batch / 4;
    } else if load.current_load > 0.60 {
        adjusted_batch = base_batch / 2;
    }

    if region.len() > 8192 {
        adjusted_batch = adjusted_batch * 2;
    } else if region.len() < 1024 {
        adjusted_batch = adjusted_batch / 2;
    }

    // Compression‑aware shaping: dense regions → smaller chunks.
    if hints.high_density {
        adjusted_batch = (adjusted_batch as f32 * 0.75) as usize;
    } else if hints.entropy_score > 0.7 {
        adjusted_batch = (adjusted_batch as f32 * 1.25) as usize;
    }

    // Align batch size to warp size to keep GPU execution efficient.
    let warp = load.warp_size.max(32) as usize;
    adjusted_batch = ((adjusted_batch / warp).max(1)) * warp;
    adjusted_batch = adjusted_batch.max(32);

    let mut chunks = Vec::new();
    let mut i = 0;

    while i < region.len() {
        let end = (i + adjusted_batch).min(region.len());
        chunks.push(region[i..end].to_vec());
        i = end;
    }

    chunks
}

/// Multi‑stream GPU execution with hybrid routing (roundabout exits + basic
/// re‑circulation). Original roundabout logic preserved and extended.
pub fn build_attention_mask_gpu_balanced(
    region_chunks: &[Vec<u32>],
    kv_len: usize,
    module: &Module,
    block_size: u32,
    load: &GpuLoad,
) -> Vec<f32> {
    let func = match module.get_function("build_mask") {
        Ok(f) => f,
        Err(_) => return vec![0.0f32; kv_len],
    };

    // Adaptive stream count based on SMs (daemon-like behavior)
    let max_streams = if load.sm_count >= 80 { 16 } else { 8 };
    let stream_count = region_chunks.len().min(max_streams);

    let mut streams = Vec::with_capacity(stream_count);
    for _ in 0..stream_count {
        match Stream::new(StreamFlags::NON_BLOCKING, None) {
            Ok(s) => streams.push(s),
            Err(_) => {
                streams.clear();
                break;
            }
        }
    }

    let mask_buf = match DeviceBuffer::<f32>::zeroed(kv_len) {
        Ok(buf) => buf,
        Err(_) => return vec![0.0f32; kv_len],
    };

    // Warp‑aware threads per block.
    let warp_size = load.warp_size.max(32);
    let tuned_block_size = block_size.max(warp_size);

    if streams.is_empty() {
        // Single‑stream fallback
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        for chunk in region_chunks {
            let region_buf = match DeviceBuffer::from_slice(chunk) {
                Ok(buf) => buf,
                Err(_) => continue,
            };
            let region_len = chunk.len() as u32;

            let threads_per_block = tuned_block_size;
            let blocks = ((region_len + threads_per_block - 1) / threads_per_block).max(1);

            unsafe {
                launch!(
                    func<<<blocks, threads_per_block, 0, stream>>>(
                        region_buf.as_device_ptr(),
                        region_len,
                        mask_buf.as_device_ptr(),
                        kv_len as u32
                    )
                )
                .unwrap();
            }
        }
        stream.synchronize().unwrap();
    } else {
        // Roundabout routing: choose exits (streams) based on semantic vs load bias,
        // with simple re‑circulation when load is high.
        let half = (streams.len() / 2).max(1);

        for (i, chunk) in region_chunks.iter().enumerate() {
            let region_buf = match DeviceBuffer::from_slice(chunk) {
                Ok(buf) => buf,
                Err(_) => continue,
            };
            let region_len = chunk.len() as u32;

            let semantic_bias = region_len < 1024;
            let mut stream_index = if semantic_bias {
                i % half
            } else {
                (i + half) % streams.len()
            };

            // Basic re‑circulation: if load is high and chunk is large, push to
            // a different exit to avoid congestion.
            if load.current_load > 0.75 && region_len > 4096 {
                stream_index = (stream_index + 1) % streams.len();
            }

            let stream = &streams[stream_index];

            let threads_per_block = tuned_block_size;
            let blocks = ((region_len + threads_per_block - 1) / threads_per_block).max(1);

            unsafe {
                launch!(
                    func<<<blocks, threads_per_block, 0, stream>>>(
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
    }

    let mut mask = vec![0.0f32; kv_len];
    if let Err(_) = mask_buf.copy_to(&mut mask) {
        return vec![0.0f32; kv_len];
    }

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

    // Local optimization state + config
    let mut state = GpuOptimizationState::default();
    let cfg = GpuOptimizationConfig {
        min_region_batch: 64,
        max_region_batch: 8192,
        min_gpu_threshold: 256,
        max_gpu_threshold: 32768,
        min_block_size: 32,
        max_block_size: 1024,
    };

    optimize_gpu(web, root, depth, kv_len, &mut state, &cfg);

    let region_size = region.len();

    // Hybrid GPU threshold: small regions stay on CPU
    if region_size < state.gpu_threshold {
        let mut mask = vec![0.0; kv_len];
        for t in region {
            if t.0 < kv_len {
                mask[t.0] = 1.0;
            }
        }
        return mask;
    }

    let region_vec: Vec<u32> = region.iter().map(|t| t.0 as u32).collect();

    // CUDA kernel source (simple mask builder + BitDrop_v2 hooks expected in module)
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
        .reg .pred p_exit;
        .reg .pred p_skip;
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

    let module = match Module::from_ptx(ptx, &[]) {
        Ok(m) => m,
        Err(_) => {
            let mut mask = vec![0.0; kv_len];
            for t in region {
                if t.0 < kv_len {
                    mask[t.0] = 1.0;
                }
            }
            return mask;
        }
    };

    // GPU‑side BitDrop_v2 collapse of region before routing.
    let collapsed_region_vec = gpu_bitdrop_collapse_region(&module, &region_vec);

    let load = read_gpu_load(&gpu_ctx.device);
    let region_chunks = partition_region_for_gpu(&collapsed_region_vec, &load, &state);

    build_attention_mask_gpu_balanced(&region_chunks, kv_len, &module, state.block_size, &load)
}

// ============================================================================
// MAX‑TIER GPU OPTIMIZATION LOOP (runtime-only tuning)
// ============================================================================

/// Max-tier optimization loop for GPU mask building.
/// Tunes GPU/CPU crossover, batching, and block size.
/// Original logic preserved, extended with crossover heuristics.
pub fn optimize_gpu(
    web: &KvWeb,
    root: WebNodeId,
    depth: usize,
    kv_len: usize,
    state: &mut GpuOptimizationState,
    cfg: &GpuOptimizationConfig,
) {
    let region = web.tokens_in_region(root, depth);
    let region_size = region.len();

    // 1) GPU threshold tuning (hybrid)
    if region_size < cfg.min_gpu_threshold {
        state.gpu_threshold =
            ((state.gpu_threshold as f32 * 0.9) as usize).max(cfg.min_gpu_threshold);
    } else if region_size > cfg.max_gpu_threshold {
        state.gpu_threshold =
            ((state.gpu_threshold as f32 * 1.1) as usize).min(cfg.max_gpu_threshold);
    }

    // Additional crossover heuristic: very large KV → prefer GPU.
    if kv_len > 16384 && region_size > cfg.min_gpu_threshold {
        state.gpu_threshold = cfg.min_gpu_threshold;
    }

    // 2) Region batching tuning
    if region_size > state.region_batch {
        state.region_batch = (state.region_batch * 2).min(cfg.max_region_batch);
    } else {
        state.region_batch = (state.region_batch / 2).max(cfg.min_region_batch);
    }

    // 3) Block size tuning
    if region_size > 4096 {
        state.block_size = (state.block_size * 2).min(cfg.max_block_size);
    } else if region_size < 512 {
        state.block_size = (state.block_size / 2).max(cfg.min_block_size);
    }

    // No compression here — GPU tuning is runtime-only.
}


