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
use cust::device::Device;
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
    /// Initialize CUDA and create a context.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        cust::init(cust::CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let ctx = Context::new(device)?; // cust 0.3.2 API
        Ok(Self { device, ctx })
    }
}

/// GPU‑accelerated attention mask builder.
/// CPU fallback if GPU unavailable.
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

    // GPU path
    let _gpu = gpu.unwrap();

    // Prepare input buffer
    let region_vec: Vec<u32> = region.iter().map(|t| t.0 as u32).collect();
    let region_len = region_vec.len() as u32;

    // Allocate GPU buffers
    let region_buf = DeviceBuffer::from_slice(&region_vec).unwrap();
    let mask_buf = DeviceBuffer::<f32>::zeroed(kv_len).unwrap();

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

        // Each thread handles one region token
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

    // Load module + kernel
    let module = Module::from_ptx(ptx, &[]).unwrap();
    let func = module.get_function("build_mask").unwrap();

    // cust 0.3.2: create a stream via Stream::new
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

    unsafe {
        launch!(
            func<<<region_len as u32, 1, 0, stream>>>(
                region_buf.as_device_ptr(),
                region_len,
                mask_buf.as_device_ptr(),
                kv_len as u32
            )
        )
        .unwrap();
    }

    // Copy result back
    let mut mask = vec![0.0f32; kv_len];
    mask_buf.copy_to(&mut mask).unwrap();

    mask
}
