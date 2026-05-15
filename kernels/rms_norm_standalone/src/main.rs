//! RMS-Norm correctness + bench binary — unified cuda-oxide compilation.
//!
//! Build and run:
//!   cargo oxide run --arch sm_80
//!   cargo oxide run --arch sm_80 -- --bench --batch 2048 --hidden-dim 4096

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, thread, warp};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use std::fs::File;

const FIXTURE: &str = "../cuda_oxide/tests/fixtures/rms_norm_4x4096.npz";
const KERNEL_EPS: f32 = 1e-5;
const MAX_ABS_ERR: f32 = 1e-4;

// =============================================================================
// CLI ARGUMENTS
// =============================================================================

struct Args {
    batch: Option<usize>,
    hidden_dim: Option<usize>,
    bench: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args { batch: None, hidden_dim: None, bench: false };
    let raw: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--batch" => {
                i += 1;
                let v = raw.get(i).ok_or("--batch requires a value")?;
                args.batch = Some(v.parse().map_err(|_| format!("--batch: {v}"))?);
            }
            "--hidden-dim" => {
                i += 1;
                let v = raw.get(i).ok_or("--hidden-dim requires a value")?;
                args.hidden_dim = Some(v.parse().map_err(|_| format!("--hidden-dim: {v}"))?);
            }
            "--bench" => args.bench = true,
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    Ok(args)
}

// =============================================================================
// DEVICE HELPER — compiled to PTX alongside the kernel
// =============================================================================

/// Approximates 1/sqrt(x) using Carmack's fast inverse square root + two
/// Newton-Raphson steps. Relative error ≈ 4.7e-7 — well inside the 1e-4
/// absolute threshold.
///
/// Uses only integer bitcast (`to_bits`/`from_bits`) and f32 mul/add.
/// `f32::sqrt()` calls `core::intrinsics::sqrtf32` which the codegen routes
/// to libdevice (`__nv_sqrtf`), forcing NVVM IR output and breaking the
/// embedded-PTX path. This helper avoids that entirely.
#[device]
fn rsqrt_approx(x: f32) -> f32 {
    let xhalf = 0.5_f32 * x;
    // Carmack magic-constant initial guess for 1/sqrt(x)
    let i = 0x5f3759df_u32.wrapping_sub(x.to_bits() >> 1);
    let y = f32::from_bits(i);
    // Two Newton-Raphson refinements: y_{n+1} = y_n * (1.5 - xhalf * y_n²)
    let y = y * (1.5_f32 - xhalf * y * y);
    y * (1.5_f32 - xhalf * y * y)
}

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

        let mut sum_sq: f32 = 0.0;
        let mut col = tid;
        while col < hdim {
            let x = input[row_base + col];
            sum_sq += x * x;
            col += 128;
        }

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
            let rms_inv = rsqrt_approx(total / hidden_dim as f32 + eps);
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
    let args = parse_args().map_err(|e| format!("arg error: {e}"))?;

    // Load fixture — correctness check always runs as a gate before bench.
    let f = File::open(FIXTURE).map_err(|e| format!("fixture not found at {FIXTURE}: {e}"))?;
    let mut npz = NpzReader::new(f)?;
    let input: Array2<f32> = npz.by_name("input.npy")?;
    let weight: Array1<f32> = npz.by_name("weight.npy")?;
    let reference: Array2<f32> = npz.by_name("output.npy")?;

    let fix_batch = input.nrows();
    let fix_hidden = input.ncols();
    let (input_vec, _) = input.into_raw_vec_and_offset();
    let (weight_vec, _) = weight.into_raw_vec_and_offset();
    let (reference_vec, _) = reference.into_raw_vec_and_offset();

    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e}"))?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).map_err(|e| format!("PTX load: {e}"))?;

    let input_dev = DeviceBuffer::from_host(&stream, &input_vec).map_err(|e| format!("alloc: {e}"))?;
    let weight_dev = DeviceBuffer::from_host(&stream, &weight_vec).map_err(|e| format!("alloc: {e}"))?;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, input_vec.len()).map_err(|e| format!("alloc: {e}"))?;

    let fix_cfg = LaunchConfig { grid_dim: (fix_batch as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    module.rms_norm_kernel(&stream, fix_cfg, &input_dev, &weight_dev, &mut output_dev, KERNEL_EPS, fix_hidden as u32)
        .map_err(|e| format!("kernel: {e}"))?;

    let output_vec = output_dev.to_host_vec(&stream).map_err(|e| format!("device→host: {e}"))?;
    let max_err = output_vec.iter().zip(&reference_vec).map(|(g, r)| (g - r).abs()).fold(0.0_f32, f32::max);

    println!("max absolute error: {max_err:.2e}");
    if max_err >= MAX_ABS_ERR {
        eprintln!("FAIL: {max_err:.2e} >= threshold {MAX_ABS_ERR:.1e}");
        std::process::exit(1);
    }
    println!("PASS (threshold {MAX_ABS_ERR:.1e})");

    if !args.bench {
        return Ok(());
    }

    // ── Bench mode ────────────────────────────────────────────────────────────
    let bench_batch = args.batch.unwrap_or(2048);
    let bench_hidden = args.hidden_dim.unwrap_or(4096);
    let n = bench_batch * bench_hidden;

    // Deterministic synthetic data; values span ±1 without an external RNG dep.
    let bench_input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001_f32).sin()).collect();
    let bench_weight: Vec<f32> =
        (0..bench_hidden).map(|i| 1.0_f32 + ((i as f32) * 0.001_f32).cos() * 0.1_f32).collect();

    let bench_in_dev = DeviceBuffer::from_host(&stream, &bench_input).map_err(|e| format!("alloc: {e}"))?;
    let bench_wt_dev = DeviceBuffer::from_host(&stream, &bench_weight).map_err(|e| format!("alloc: {e}"))?;
    let mut bench_out_dev = DeviceBuffer::<f32>::zeroed(&stream, n).map_err(|e| format!("alloc: {e}"))?;
    // 1-element buffer used as a stream-sync barrier; D2H of 4 bytes ≈ zero overhead.
    let sync_buf = DeviceBuffer::<f32>::zeroed(&stream, 1).map_err(|e| format!("alloc: {e}"))?;

    let bench_cfg = LaunchConfig { grid_dim: (bench_batch as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };

    for _ in 0..100 {
        module.rms_norm_kernel(&stream, bench_cfg, &bench_in_dev, &bench_wt_dev, &mut bench_out_dev, KERNEL_EPS, bench_hidden as u32)
            .map_err(|e| format!("warm-up: {e}"))?;
    }
    sync_buf.to_host_vec(&stream).map_err(|e| format!("sync: {e}"))?;

    let mut times_us: Vec<f64> = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let t0 = std::time::Instant::now();
        module.rms_norm_kernel(&stream, bench_cfg, &bench_in_dev, &bench_wt_dev, &mut bench_out_dev, KERNEL_EPS, bench_hidden as u32)
            .map_err(|e| format!("bench: {e}"))?;
        sync_buf.to_host_vec(&stream).map_err(|e| format!("sync: {e}"))?;
        times_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = times_us[499];
    let p99 = times_us[989];
    let bytes = (2 * bench_batch * bench_hidden + bench_hidden) as f64 * 4.0;
    let throughput_gbs = bytes / (p50 / 1e6) / 1e9;

    println!(
        r#"{{"backend":"cuda_oxide","batch":{b},"hidden_dim":{h},"p50_us":{p50:.3},"p99_us":{p99:.3},"throughput_gbs":{gbs:.2}}}"#,
        b = bench_batch, h = bench_hidden, p50 = p50, p99 = p99, gbs = throughput_gbs
    );

    Ok(())
}
