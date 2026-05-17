//! RMS Normalization kernel (sm_80, A100).
//!
//! DEPRECATED: This file uses the split cfg/PTX-file pattern. The canonical
//! implementation is `src/bin/rms_norm.rs`, which uses unified cuda-oxide
//! compilation (PTX embedded at build time, no env-var, no cfg splits).
//!
//! Implements: `y = x / sqrt(mean(x²) + eps) * weight`
//!
//! References:
//! - RMSNorm paper: <https://arxiv.org/abs/1910.07467>
//! - Triton baseline: `kernels/triton/rms_norm.py`

use thiserror::Error;

#[cfg(feature = "gpu")]
use cuda_device::{cuda_module, kernel, thread, warp, DisjointSlice, SharedArray};

#[derive(Debug, Error)]
pub enum RmsNormError {
    #[error("input tensor rank must be 2, got {0}")]
    InvalidRank(usize),
    #[error("CUDA kernel launch failed: {0}")]
    CudaLaunch(String),
}

/// RMS Normalization kernel wrapper (sm_80).
pub struct RmsNorm {
    eps: f32,
}

/// Device kernels compiled to PTX by `cargo oxide build --features gpu`.
///
/// Grid  : `(batch_size, 1, 1)` — one block per input row.
/// Block : `(128, 1, 1)` — four warps; each thread covers `hidden_dim / 128` elements.
#[cfg(feature = "gpu")]
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Scalar-naive RMS-Norm kernel.
    ///
    /// Passes:
    ///   1. Each thread accumulates sum-of-squares over its strided column elements.
    ///   2. Warp shuffle-down reduces 32 lanes → lane 0 holds the warp partial.
    ///   3. Lane 0 of each warp stores its partial to `PARTIAL[warp_id]`.
    ///   4. Thread 0 sums the 4 partials, computes `rms_inv`, stores to `RMS_INV[0]`.
    ///   5. After sync, each thread reads the broadcast `rms_inv` and writes scaled outputs.
    #[kernel]
    pub fn rms_norm_kernel(
        input: &[f32],
        weight: &[f32],
        mut output: DisjointSlice<f32>,
        eps: f32,
        hidden_dim: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 4> = SharedArray::UNINIT;
        static mut RMS_INV: SharedArray<f32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize;
        let hdim = hidden_dim as usize;
        let row_base = row * hdim;

        // Pass 1: accumulate sum-of-squares over this thread's strided column elements.
        let mut sum_sq: f32 = 0.0;
        let mut col = tid;
        while col < hdim {
            let x = input[row_base + col];
            sum_sq += x * x;
            col += 128;
        }

        // Warp-level reduction: five shuffle-downs fold 32 lanes into lane 0.
        sum_sq += warp::shuffle_down_f32(sum_sq, 16);
        sum_sq += warp::shuffle_down_f32(sum_sq, 8);
        sum_sq += warp::shuffle_down_f32(sum_sq, 4);
        sum_sq += warp::shuffle_down_f32(sum_sq, 2);
        sum_sq += warp::shuffle_down_f32(sum_sq, 1);

        if warp::lane_id() == 0 {
            let warp_id = tid / 32;
            unsafe {
                // SAFETY: only lane 0 of each warp executes this write; warp_id ∈ 0..4
                // (128 threads / 32 per warp), so each warp writes to a distinct slot.
                PARTIAL[warp_id] = sum_sq;
            }
        }

        thread::sync_threads();

        if tid == 0 {
            let total = unsafe {
                // SAFETY: sync_threads() above ensures all four warp lane-0 writes to
                // PARTIAL are globally visible before this read.
                PARTIAL[0] + PARTIAL[1] + PARTIAL[2] + PARTIAL[3]
            };
            let rms_inv = 1.0_f32 / (total / hidden_dim as f32 + eps).sqrt();
            unsafe {
                // SAFETY: only thread 0 writes RMS_INV[0]; all other threads read it
                // after the next sync_threads().
                RMS_INV[0] = rms_inv;
            }
        }

        thread::sync_threads();

        let rms_inv = unsafe {
            // SAFETY: sync_threads() above guarantees thread 0's write to RMS_INV[0]
            // is visible to every thread in the block before this read.
            RMS_INV[0]
        };

        // Pass 2: normalize and write output.
        let mut col = tid;
        while col < hdim {
            let out_idx = row_base + col;
            let y = input[out_idx] * rms_inv * weight[col];
            unsafe {
                // SAFETY: out_idx = row * hdim + col. blockIdx.x is unique per block,
                // so `row` is unique per block. Within a block col ≡ tid (mod 128), and
                // each thread has a unique tid ∈ 0..128, so every thread in the grid
                // writes to a distinct out_idx — no concurrent writes to the same slot.
                *output.get_unchecked_mut(out_idx) = y;
            }
            col += 128;
        }
    }
}

impl RmsNorm {
    /// Create a new RmsNorm kernel with the given epsilon for numerical stability.
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }

    /// Launch the RMS norm kernel on the given input.
    ///
    /// # Arguments
    /// * `input`  — row-major f32 tensor of shape `[batch, hidden_dim]`
    /// * `weight` — learnable scale parameter of shape `[hidden_dim]`
    ///
    /// # Errors
    ///
    /// Returns [`RmsNormError::CudaLaunch`] if the CUDA kernel fails to launch.
    #[cfg(feature = "gpu")]
    pub fn forward(&self, input: &[f32], weight: &[f32]) -> Result<Vec<f32>, RmsNormError> {
        use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
        use std::sync::Arc;

        let hidden_dim = weight.len();
        let batch = input.len() / hidden_dim;

        let ctx: Arc<CudaContext> =
            CudaContext::new(0).map_err(|e| RmsNormError::CudaLaunch(e.to_string()))?;
        let stream = ctx.default_stream();

        // PTX path: set KERNELSERVE_PTX or place the file at ptx/kernelserve-kernels.ptx
        let ptx_path = std::env::var("KERNELSERVE_PTX")
            .unwrap_or_else(|_| "ptx/kernelserve-kernels.ptx".to_string());
        let cu_module = ctx
            .load_module_from_file(&ptx_path)
            .map_err(|e| RmsNormError::CudaLaunch(e.to_string()))?;
        let module =
            kernels::from_module(cu_module).map_err(|e| RmsNormError::CudaLaunch(e.to_string()))?;

        let input_dev = DeviceBuffer::from_host(&stream, input)
            .map_err(|e| RmsNormError::CudaLaunch(e.to_string()))?;
        let weight_dev = DeviceBuffer::from_host(&stream, weight)
            .map_err(|e| RmsNormError::CudaLaunch(e.to_string()))?;
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, input.len())
            .map_err(|e| RmsNormError::CudaLaunch(e.to_string()))?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        module
            .rms_norm_kernel(
                &stream,
                cfg,
                &input_dev,
                &weight_dev,
                &mut output_dev,
                self.eps,
                hidden_dim as u32,
            )
            .map_err(|e| RmsNormError::CudaLaunch(e.to_string()))?;

        output_dev
            .to_host_vec(&stream)
            .map_err(|e| RmsNormError::CudaLaunch(e.to_string()))
    }

    #[cfg(not(feature = "gpu"))]
    pub fn forward(&self, input: &[f32], weight: &[f32]) -> Result<Vec<f32>, RmsNormError> {
        let hidden_dim = weight.len();
        let batch = input.len() / hidden_dim;
        Ok(crate::cpu_ref::rms_norm_cpu(
            input, weight, batch, hidden_dim, self.eps,
        ))
    }
}
