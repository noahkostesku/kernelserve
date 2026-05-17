from __future__ import annotations

import re
import sys
from pathlib import Path

_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

# ---------------------------------------------------------------------------
# Templates — use __NAME__ (snake_case) and __PASCAL__ (PascalCase) as
# placeholders. All Rust/TOML/Python braces are literal; no .format() escaping.
# ---------------------------------------------------------------------------

_CUDA_OXIDE_RS = """\
//! __PASCAL__ kernel (sm_80, A100).
//!
//! TODO: describe the mathematical operation this kernel implements.
//!
//! References:
//! - TODO: add paper or algorithm reference
//! - Triton baseline: `kernels/triton/__NAME__.py`

use thiserror::Error;

#[cfg(feature = "gpu")]
use cuda_device::{cuda_module, kernel, thread, warp, DisjointSlice, SharedArray};

#[derive(Debug, Error)]
pub enum __PASCAL__Error {
    #[error("TODO: describe invalid-rank condition, got {0}")]
    InvalidRank(usize),
    #[error("CUDA kernel launch failed: {0}")]
    CudaLaunch(String),
}

/// __PASCAL__ kernel wrapper (sm_80).
pub struct __PASCAL__ {
    // TODO: add kernel configuration fields (e.g., eps: f32)
}

/// Device kernels compiled to PTX by `cargo oxide build --features gpu`.
///
/// Grid  : `(batch_size, 1, 1)` — one block per input row.
/// Block : `(128, 1, 1)`        — four warps.
#[cfg(feature = "gpu")]
#[cuda_module]
pub mod kernels {
    use super::*;

    /// TODO: describe what this kernel computes, its reduction passes,
    /// and shared-memory layout.
    #[kernel]
    pub fn __NAME___kernel(
        input: &[f32],
        // TODO: add remaining kernel parameters
        mut output: DisjointSlice<f32>,
        hidden_dim: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 4> = SharedArray::UNINIT;
        static mut OUT_VAL: SharedArray<f32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize;
        let hdim = hidden_dim as usize;
        let row_base = row * hdim;

        // TODO: Pass 1 — accumulate per-thread partial across strided columns.
        let mut acc: f32 = 0.0;
        let mut col = tid;
        while col < hdim {
            acc += input[row_base + col]; // TODO: replace with actual accumulation
            col += 128;
        }

        // Warp-level reduction: five shuffle-downs fold 32 lanes into lane 0.
        acc += warp::shuffle_down_f32(acc, 16);
        acc += warp::shuffle_down_f32(acc, 8);
        acc += warp::shuffle_down_f32(acc, 4);
        acc += warp::shuffle_down_f32(acc, 2);
        acc += warp::shuffle_down_f32(acc, 1);

        if warp::lane_id() == 0 {
            let warp_id = tid / 32;
            unsafe {
                // SAFETY: only lane 0 of each warp executes this write; warp_id ∈ 0..4
                // (128 threads / 32 per warp), so each warp writes to a distinct slot.
                PARTIAL[warp_id] = acc;
            }
        }

        thread::sync_threads();

        if tid == 0 {
            let total = unsafe {
                // SAFETY: sync_threads() above ensures all four warp lane-0 writes to
                // PARTIAL are globally visible before this read.
                PARTIAL[0] + PARTIAL[1] + PARTIAL[2] + PARTIAL[3]
            };
            let broadcast_val = total; // TODO: compute the scalar to broadcast
            unsafe {
                // SAFETY: only thread 0 writes OUT_VAL[0]; all other threads read it
                // after the next sync_threads().
                OUT_VAL[0] = broadcast_val;
            }
        }

        thread::sync_threads();

        let broadcast_val = unsafe {
            // SAFETY: sync_threads() above guarantees thread 0's write to OUT_VAL[0]
            // is visible to every thread in the block before this read.
            OUT_VAL[0]
        };

        // TODO: Pass 2 — write output using broadcast_val.
        let mut col = tid;
        while col < hdim {
            let out_idx = row_base + col;
            let y = input[out_idx] * broadcast_val; // TODO: replace with actual formula
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

impl __PASCAL__ {
    pub fn new() -> Self {
        // TODO: accept configuration parameters
        Self {}
    }

    /// Launch the __NAME__ kernel on the given input.
    ///
    /// # Errors
    ///
    /// Returns [`__PASCAL__Error::CudaLaunch`] if the CUDA kernel fails to launch.
    #[cfg(feature = "gpu")]
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, __PASCAL__Error> {
        use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
        use std::sync::Arc;

        // TODO: derive dimensions from input
        let hidden_dim: usize = 0; // placeholder — must be set before launch
        let batch = input.len() / hidden_dim.max(1);

        let ctx: Arc<CudaContext> =
            CudaContext::new(0).map_err(|e| __PASCAL__Error::CudaLaunch(e.to_string()))?;
        let stream = ctx.default_stream();

        // PTX path: set KERNELSERVE_PTX or place the file at ptx/kernelserve-kernels.ptx
        let ptx_path = std::env::var("KERNELSERVE_PTX")
            .unwrap_or_else(|_| "ptx/kernelserve-kernels.ptx".to_string());
        let cu_module = ctx
            .load_module_from_file(&ptx_path)
            .map_err(|e| __PASCAL__Error::CudaLaunch(e.to_string()))?;
        let module =
            kernels::from_module(cu_module)
                .map_err(|e| __PASCAL__Error::CudaLaunch(e.to_string()))?;

        let input_dev = DeviceBuffer::from_host(&stream, input)
            .map_err(|e| __PASCAL__Error::CudaLaunch(e.to_string()))?;
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, input.len())
            .map_err(|e| __PASCAL__Error::CudaLaunch(e.to_string()))?;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        // TODO: pass correct arguments matching __NAME___kernel signature
        module
            .__NAME___kernel(&stream, cfg, &input_dev, &mut output_dev, hidden_dim as u32)
            .map_err(|e| __PASCAL__Error::CudaLaunch(e.to_string()))?;

        output_dev
            .to_host_vec(&stream)
            .map_err(|e| __PASCAL__Error::CudaLaunch(e.to_string()))
    }

    #[cfg(not(feature = "gpu"))]
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, __PASCAL__Error> {
        // TODO: implement CPU fallback (or delegate to crate::cpu_ref)
        let _ = input;
        todo!("__NAME__ CPU fallback not yet implemented")
    }
}
"""

_TRITON_PY = """\
# Triton baseline for __PASCAL__.
#
# This is the reference implementation used for:
# 1. Correctness validation against kernels/cuda_oxide/src/__NAME__.rs
# 2. Performance comparison via `ks compare --kernel __NAME__`
#
# PyTorch reference: __NAME___torch() below
# Rust counterpart:  kernels/cuda_oxide/src/__NAME__.rs

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# triton is Linux+CUDA only; CPU mock mode works on macOS without it.
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def ___NAME___kernel(
        x_ptr,
        out_ptr,
        stride_row,
        hidden_dim,
        # TODO: add kernel-specific constexpr and scalar parameters
        BLOCK_SIZE: tl.constexpr,
    ):
        # TODO: implement the Triton kernel body.
        # One program per input row:
        pid = tl.program_id(axis=0)
        row_start = pid * stride_row
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_dim

        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)

        # TODO: compute output from x
        y = x  # placeholder: replace with actual computation

        tl.store(out_ptr + row_start + offs, y, mask=mask)


def __NAME___triton(x: torch.Tensor, **kwargs) -> torch.Tensor:
    \"\"\"Launch the Triton __PASCAL__ kernel; falls back to __NAME___torch when unavailable.

    Args:
        x:       Input tensor of shape [batch, hidden_dim], fp32, CUDA
        **kwargs: TODO: document kernel-specific keyword arguments

    Returns:
        Output tensor of the same shape as x
    \"\"\"
    if not _TRITON_AVAILABLE:
        return __NAME___torch(x, **kwargs)

    batch, hidden_dim = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    ___NAME___kernel[(batch,)](
        x,
        out,
        x.stride(0),
        hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def __NAME___torch(x: torch.Tensor, **kwargs) -> torch.Tensor:
    \"\"\"PyTorch reference implementation (used for correctness checks and fixture generation).

    TODO: implement the reference operation using torch ops.
    \"\"\"
    # TODO: replace with actual PyTorch computation
    return x.clone()


def generate_fixture(batch: int, hidden_dim: int, seed: int = 42) -> Path:
    \"\"\"Generate a deterministic test fixture and write it to tests/fixtures/.

    The fixture is a numpy .npz archive with keys:
        input   — shape [batch, hidden_dim], fp32
        output  — shape [batch, hidden_dim], fp32  (PyTorch reference)

    Returns the path written.
    \"\"\"
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn(batch, hidden_dim, generator=gen)
    output = __NAME___torch(x)  # TODO: pass required kwargs

    fixtures_dir = Path(__file__).resolve().parent.parent / "cuda_oxide" / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    out_path = fixtures_dir / f"__NAME___{batch}x{hidden_dim}.npz"
    np.savez(
        out_path,
        input=x.numpy(),
        output=output.numpy(),
    )
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="__PASCAL__ Triton baseline utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen_p = sub.add_parser("generate", help="Generate a correctness fixture (.npz)")
    gen_p.add_argument("--batch", type=int, default=4)
    gen_p.add_argument("--hidden_dim", type=int, default=4096)
    gen_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "generate":
        path = generate_fixture(args.batch, args.hidden_dim, args.seed)
        print(f"wrote {path}")
"""

_STANDALONE_RS = """\
//! __PASCAL__ correctness + bench binary — unified cuda-oxide compilation.
//!
//! Build and run:
//!   cargo oxide run --arch sm_80
//!   cargo oxide run --arch sm_80 -- --bench --batch 2048 --hidden-dim 4096

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, thread, warp};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use std::fs::File;

// TODO: update fixture path once you run `python -m kernels.triton.__NAME__ generate`
const FIXTURE: &str = "../cuda_oxide/tests/fixtures/__NAME___4x4096.npz";
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
/// TODO: keep or remove depending on whether your kernel needs rsqrt.
#[device]
fn rsqrt_approx(x: f32) -> f32 {
    let xhalf = 0.5_f32 * x;
    let i = 0x5f3759df_u32.wrapping_sub(x.to_bits() >> 1);
    let y = f32::from_bits(i);
    let y = y * (1.5_f32 - xhalf * y * y);
    y * (1.5_f32 - xhalf * y * y)
}

// =============================================================================
// KERNEL — compiled to PTX by rustc-codegen-cuda
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// TODO: describe this kernel, its grid/block layout, and reduction passes.
    ///
    /// Grid  : (batch_size, 1, 1) — one block per input row.
    /// Block : (128, 1, 1)        — four warps.
    #[kernel]
    pub fn __NAME___kernel(
        input: &[f32],
        // TODO: add remaining parameters matching cuda_oxide/src/__NAME__.rs
        mut output: DisjointSlice<f32>,
        hidden_dim: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 4> = SharedArray::UNINIT;
        static mut OUT_VAL: SharedArray<f32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize;
        let hdim = hidden_dim as usize;
        let row_base = row * hdim;

        // TODO: Pass 1 — accumulate per-thread partial across strided columns.
        let mut acc: f32 = 0.0;
        let mut col = tid;
        while col < hdim {
            acc += input[row_base + col]; // TODO: replace with actual accumulation
            col += 128;
        }

        // Warp-level reduction: five shuffle-downs fold 32 lanes into lane 0.
        acc += warp::shuffle_down_f32(acc, 16);
        acc += warp::shuffle_down_f32(acc, 8);
        acc += warp::shuffle_down_f32(acc, 4);
        acc += warp::shuffle_down_f32(acc, 2);
        acc += warp::shuffle_down_f32(acc, 1);

        if warp::lane_id() == 0 {
            let warp_id = tid / 32;
            unsafe {
                // SAFETY: only lane 0 of each warp executes this write; warp_id ∈ 0..4
                // (128 threads / 32 per warp), so each warp writes to a distinct slot.
                PARTIAL[warp_id] = acc;
            }
        }

        thread::sync_threads();

        if tid == 0 {
            let total = unsafe {
                // SAFETY: sync_threads() above ensures all four warp lane-0 writes to
                // PARTIAL are globally visible before this read.
                PARTIAL[0] + PARTIAL[1] + PARTIAL[2] + PARTIAL[3]
            };
            let broadcast_val = total; // TODO: compute the scalar to broadcast
            unsafe {
                // SAFETY: only thread 0 writes OUT_VAL[0]; all other threads read it
                // after the next sync_threads().
                OUT_VAL[0] = broadcast_val;
            }
        }

        thread::sync_threads();

        let broadcast_val = unsafe {
            // SAFETY: sync_threads() above guarantees thread 0's write to OUT_VAL[0]
            // is visible to every thread in the block before this read.
            OUT_VAL[0]
        };

        // TODO: Pass 2 — write output using broadcast_val.
        let mut col = tid;
        while col < hdim {
            let out_idx = row_base + col;
            let y = input[out_idx] * broadcast_val; // TODO: replace with actual formula
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

/// CPU reference for correctness checks when the fixture shape doesn't match.
/// TODO: implement to match the PyTorch reference in kernels/triton/__NAME__.py
fn __NAME___cpu(input: &[f32], batch: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * hidden];
    for row in 0..batch {
        let base = row * hidden;
        // TODO: implement CPU reference computation
        out[base..base + hidden].copy_from_slice(&input[base..base + hidden]);
    }
    out
}

fn check_and_print(max_err: f32) {
    println!("max absolute error: {max_err:.2e}");
    if max_err >= MAX_ABS_ERR {
        eprintln!("FAIL: {max_err:.2e} >= threshold {MAX_ABS_ERR:.1e}");
        std::process::exit(1);
    }
    println!("PASS (threshold {MAX_ABS_ERR:.1e})");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("arg error: {e}"))?;

    let custom_dims = args.batch.is_some() || args.hidden_dim.is_some();

    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e}"))?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).map_err(|e| format!("PTX load: {e}"))?;

    if !custom_dims {
        let f = File::open(FIXTURE).map_err(|e| format!("fixture not found at {FIXTURE}: {e}"))?;
        let mut npz = NpzReader::new(f)?;
        let input: Array2<f32> = npz.by_name("input.npy")?;
        // TODO: load additional fixture arrays as needed (e.g. weight)
        let reference: Array2<f32> = npz.by_name("output.npy")?;

        let fix_batch = input.nrows();
        let fix_hidden = input.ncols();
        let (input_vec, _) = input.into_raw_vec_and_offset();
        let (reference_vec, _) = reference.into_raw_vec_and_offset();

        let input_dev = DeviceBuffer::from_host(&stream, &input_vec)
            .map_err(|e| format!("alloc: {e}"))?;
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, input_vec.len())
            .map_err(|e| format!("alloc: {e}"))?;

        let fix_cfg = LaunchConfig {
            grid_dim: (fix_batch as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // TODO: pass correct arguments matching __NAME___kernel signature
        module
            .__NAME___kernel(&stream, fix_cfg, &input_dev, &mut output_dev, fix_hidden as u32)
            .map_err(|e| format!("kernel: {e}"))?;

        let output_vec = output_dev
            .to_host_vec(&stream)
            .map_err(|e| format!("device→host: {e}"))?;
        let max_err = output_vec
            .iter()
            .zip(&reference_vec)
            .map(|(g, r)| (g - r).abs())
            .fold(0.0_f32, f32::max);
        check_and_print(max_err);

        if !args.bench {
            return Ok(());
        }
    }

    let bench_batch = args.batch.unwrap_or(2048);
    let bench_hidden = args.hidden_dim.unwrap_or(4096);
    let n = bench_batch * bench_hidden;

    let bench_input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001_f32).sin()).collect();

    let bench_in_dev = DeviceBuffer::from_host(&stream, &bench_input)
        .map_err(|e| format!("alloc: {e}"))?;
    let mut bench_out_dev = DeviceBuffer::<f32>::zeroed(&stream, n)
        .map_err(|e| format!("alloc: {e}"))?;

    let bench_cfg = LaunchConfig {
        grid_dim: (bench_batch as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    if custom_dims {
        let cpu_ref = __NAME___cpu(&bench_input, bench_batch, bench_hidden);
        module
            .__NAME___kernel(
                &stream, bench_cfg, &bench_in_dev, &mut bench_out_dev, bench_hidden as u32,
            )
            .map_err(|e| format!("kernel: {e}"))?;
        let output_vec = bench_out_dev
            .to_host_vec(&stream)
            .map_err(|e| format!("device→host: {e}"))?;
        let max_err = output_vec
            .iter()
            .zip(&cpu_ref)
            .map(|(g, r)| (g - r).abs())
            .fold(0.0_f32, f32::max);
        check_and_print(max_err);

        if !args.bench {
            return Ok(());
        }
    }

    // 1-element buffer used as a stream-sync barrier.
    let sync_buf = DeviceBuffer::<f32>::zeroed(&stream, 1)
        .map_err(|e| format!("alloc: {e}"))?;

    for _ in 0..100 {
        module
            .__NAME___kernel(
                &stream, bench_cfg, &bench_in_dev, &mut bench_out_dev, bench_hidden as u32,
            )
            .map_err(|e| format!("warm-up: {e}"))?;
    }
    sync_buf.to_host_vec(&stream).map_err(|e| format!("sync: {e}"))?;

    let mut times_us: Vec<f64> = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let t0 = std::time::Instant::now();
        module
            .__NAME___kernel(
                &stream, bench_cfg, &bench_in_dev, &mut bench_out_dev, bench_hidden as u32,
            )
            .map_err(|e| format!("bench: {e}"))?;
        sync_buf.to_host_vec(&stream).map_err(|e| format!("sync: {e}"))?;
        times_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = times_us[499];
    let p99 = times_us[989];
    // TODO: adjust bytes formula for your kernel's actual memory access pattern
    let bytes = (2 * bench_batch * bench_hidden) as f64 * 4.0;
    let throughput_gbs = bytes / (p50 / 1e6) / 1e9;

    println!(
        r#"{{"backend":"cuda_oxide","batch":{b},"hidden_dim":{h},"p50_us":{p50:.3},"p99_us":{p99:.3},"throughput_gbs":{gbs:.2}}}"#,
        b = bench_batch, h = bench_hidden, p50 = p50, p99 = p99, gbs = throughput_gbs
    );

    Ok(())
}
"""

_CARGO_TOML = """\
[package]
name        = "__NAME__"
version     = "0.1.0"
edition     = "2021"
description = "cuda-oxide standalone correctness binary for __PASCAL__ (sm_80 A100)"
publish     = false

[workspace]  # isolate from any parent workspace

[dependencies]
cuda-device = { git = "https://github.com/NVlabs/cuda-oxide", rev = "f819f23fe035ec1c24b8ec06e738b62a9f377140" }
cuda-host   = { git = "https://github.com/NVlabs/cuda-oxide", rev = "f819f23fe035ec1c24b8ec06e738b62a9f377140" }
cuda-core   = { git = "https://github.com/NVlabs/cuda-oxide", rev = "f819f23fe035ec1c24b8ec06e738b62a9f377140" }
ndarray     = "0.16"
ndarray-npy = "0.9"

[profile.release]
opt-level     = 3
lto           = "thin"
codegen-units = 1
"""

_TEST_PY = """\
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kernels.triton.__NAME__ import __NAME___torch, __NAME___triton

FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "kernels"
    / "cuda_oxide"
    / "tests"
    / "fixtures"
    / "__NAME___4x4096.npz"
)
ATOL = 1e-4


def test___NAME___torch_vs_fixture() -> None:
    \"\"\"Validate the PyTorch reference against the pre-generated fixture.

    Run `python -m kernels.triton.__NAME__ generate` first to create the fixture.
    \"\"\"
    if not FIXTURE.exists():
        pytest.skip(
            f"fixture not found at {FIXTURE} — run: "
            f"python -m kernels.triton.__NAME__ generate"
        )
    data = np.load(FIXTURE)
    x = torch.from_numpy(data["input"])
    # TODO: load additional fixture arrays (e.g. weight) if your kernel requires them
    ref = torch.from_numpy(data["output"])
    out = __NAME___torch(x)  # TODO: pass required kwargs
    max_err = float((out - ref).abs().max())
    assert max_err < ATOL, f"max abs error {max_err:.2e} >= threshold {ATOL:.1e}"


@pytest.mark.gpu
def test___NAME___triton_vs_torch() -> None:
    \"\"\"GPU test: Triton kernel must agree with the PyTorch reference within ATOL.\"\"\"
    torch.manual_seed(42)
    x = torch.randn(4, 4096, device="cuda", dtype=torch.float32)
    # TODO: set up additional inputs (e.g. weight tensor) required by your kernel
    ref = __NAME___torch(x)  # TODO: pass required kwargs
    out = __NAME___triton(x)  # TODO: pass required kwargs
    max_err = float((out.cpu() - ref.cpu()).abs().max())
    assert max_err < ATOL, f"max abs error {max_err:.2e} >= threshold {ATOL:.1e}"
"""


def _to_pascal(name: str) -> str:
    return "".join(w.capitalize() for w in name.split("_"))


def find_project_root() -> Path:
    # __file__ is kernelserve/cli/new_kernel.py
    # .parent       → kernelserve/cli/
    # .parent.parent → kernelserve/
    # .parent.parent.parent → project root (contains kernels/, tests/, kernelserve/)
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / "kernels").is_dir():
        print(
            f"error: cannot locate project root from {__file__!r}; "
            f"expected 'kernels/' directory at {root}",
            file=sys.stderr,
        )
        sys.exit(1)
    return root


def validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        print(
            f"error: {name!r} is not a valid identifier\n"
            f"       must match ^[a-zA-Z][a-zA-Z0-9_]*$",
            file=sys.stderr,
        )
        sys.exit(1)


def _target_paths(root: Path, name: str) -> list[Path]:
    return [
        root / "kernels" / "cuda_oxide" / "src" / f"{name}.rs",
        root / "kernels" / "triton" / f"{name}.py",
        root / "kernels" / f"{name}_standalone" / "src" / "main.rs",
        root / "kernels" / f"{name}_standalone" / "Cargo.toml",
        root / "tests" / "unit" / f"test_{name}.py",
    ]


def check_conflicts(root: Path, name: str) -> None:
    conflicts = [p for p in _target_paths(root, name) if p.exists()]
    if conflicts:
        for p in conflicts:
            print(f"error: file already exists: {p.relative_to(root)}", file=sys.stderr)
        sys.exit(1)


def _expand(template: str, name: str, pascal: str) -> str:
    return template.replace("__NAME__", name).replace("__PASCAL__", pascal)


def write_files(root: Path, name: str) -> None:
    pascal = _to_pascal(name)
    files = {
        root / "kernels" / "cuda_oxide" / "src" / f"{name}.rs": _expand(
            _CUDA_OXIDE_RS, name, pascal
        ),
        root / "kernels" / "triton" / f"{name}.py": _expand(_TRITON_PY, name, pascal),
        root / "kernels" / f"{name}_standalone" / "src" / "main.rs": _expand(
            _STANDALONE_RS, name, pascal
        ),
        root / "kernels" / f"{name}_standalone" / "Cargo.toml": _expand(
            _CARGO_TOML, name, pascal
        ),
        root / "tests" / "unit" / f"test_{name}.py": _expand(_TEST_PY, name, pascal),
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def print_instructions(name: str) -> None:
    print(f"Scaffolded kernel: {name}")
    print()
    print(f"  kernels/cuda_oxide/src/{name}.rs")
    print(f"  kernels/triton/{name}.py")
    print(f"  kernels/{name}_standalone/src/main.rs")
    print(f"  kernels/{name}_standalone/Cargo.toml")
    print(f"  tests/unit/test_{name}.py")
    print()
    print("Next steps:")
    print(f"  1. Implement your kernel in kernels/cuda_oxide/src/{name}.rs")
    print(f"  2. cargo oxide build --arch sm_80")
    print(f"  3. ks compare --kernel {name}")


def run_new_kernel(args: object) -> None:
    name: str = args.name  # type: ignore[attr-defined]
    validate_name(name)
    root = find_project_root()
    check_conflicts(root, name)
    write_files(root, name)
    print_instructions(name)
