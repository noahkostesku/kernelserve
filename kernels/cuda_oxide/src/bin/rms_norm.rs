//! RMS-Norm correctness binary — unified cuda-oxide compilation.
//!
//! Kernel + host code in one file; PTX is embedded at compile time by
//! `cargo oxide`. No PTX file path, no env-var, no cfg splits.
//!
//! Build and run:
//!   cargo oxide run rms_norm

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use std::fs::File;

const FIXTURE: &str = "tests/fixtures/rms_norm_4x4096.npz";
/// Epsilon for numerical stability inside the kernel.
const KERNEL_EPS: f32 = 1e-5;
/// Correctness threshold: max abs error vs. PyTorch reference (fp32).
const MAX_ABS_ERR: f32 = 1e-4;

// =============================================================================
// KERNEL — compiled to PTX by rustc-codegen-cuda
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Scalar-naive RMS-Norm kernel.
    ///
    /// Grid  : (batch_size, 1, 1) — one block per input row.
    /// Block : (128, 1, 1)        — four warps.
    ///
    /// Passes:
    ///   1. Each thread accumulates sum-of-squares over its strided columns.
    ///   2. Warp shuffle-down reduces 32 lanes → lane 0 holds the warp partial.
    ///   3. Lane 0 of each warp stores its partial to `PARTIAL[warp_id]`.
    ///   4. Thread 0 sums the 4 partials, computes `rms_inv`, stores to `RMS_INV[0]`.
    ///   5. After sync, each thread reads the broadcast `rms_inv` and writes output.
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

        // Pass 1: accumulate sum-of-squares over this thread's strided columns.
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

// =============================================================================
// HOST CODE — compiled to native x86_64 by LLVM
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = File::open(FIXTURE)
        .map_err(|e| format!("fixture not found at {FIXTURE}: {e}"))?;
    let mut npz = NpzReader::new(f)?;

    let input: Array2<f32> = npz.by_name("input.npy")?;
    let weight: Array1<f32> = npz.by_name("weight.npy")?;
    let reference: Array2<f32> = npz.by_name("output.npy")?;

    let batch = input.nrows();
    let hidden_dim = input.ncols();
    let (input_vec, _) = input.into_raw_vec_and_offset();
    let (weight_vec, _) = weight.into_raw_vec_and_offset();
    let (reference_vec, _) = reference.into_raw_vec_and_offset();

    let ctx = CudaContext::new(0)
        .map_err(|e| format!("CUDA context failed: {e}"))?;
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx)
        .map_err(|e| format!("embedded PTX load failed: {e}"))?;

    let input_dev = DeviceBuffer::from_host(&stream, &input_vec)
        .map_err(|e| format!("device alloc failed: {e}"))?;
    let weight_dev = DeviceBuffer::from_host(&stream, &weight_vec)
        .map_err(|e| format!("device alloc failed: {e}"))?;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, input_vec.len())
        .map_err(|e| format!("device alloc failed: {e}"))?;

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
            KERNEL_EPS,
            hidden_dim as u32,
        )
        .map_err(|e| format!("kernel launch failed: {e}"))?;

    let output_vec = output_dev
        .to_host_vec(&stream)
        .map_err(|e| format!("device→host copy failed: {e}"))?;

    let max_err = output_vec
        .iter()
        .zip(&reference_vec)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0_f32, f32::max);

    println!("max absolute error: {max_err:.2e}");

    if max_err >= MAX_ABS_ERR {
        eprintln!("FAIL: {max_err:.2e} >= threshold {MAX_ABS_ERR:.1e}");
        std::process::exit(1);
    }

    println!("PASS (threshold {MAX_ABS_ERR:.1e})");
    Ok(())
}
