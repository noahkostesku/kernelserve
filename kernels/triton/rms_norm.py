# Triton baseline for RMS Normalization.
#
# This is the reference implementation used for:
# 1. Correctness validation against kernels/cuda_oxide/src/rms_norm.rs
# 2. Performance comparison in tests/benchmark/
#
# PyTorch reference: rms_norm_torch() below
# Rust counterpart:  kernels/cuda_oxide/src/rms_norm.rs

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
    def _rms_norm_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        stride_row,
        hidden_dim,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per input row.
        pid = tl.program_id(axis=0)
        row_start = pid * stride_row
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_dim

        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)

        mean_sq = tl.sum(x * x, axis=0) / hidden_dim
        rms_inv = tl.rsqrt(mean_sq + eps)

        y = x * rms_inv * w
        tl.store(out_ptr + row_start + offs, y, mask=mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Launch the Triton RMS norm kernel; falls back to rms_norm_torch when Triton is unavailable.

    Args:
        x:      Input tensor of shape [batch, hidden_dim], fp32, CUDA
        weight: Scale parameter of shape [hidden_dim], fp32, CUDA
        eps:    Epsilon for numerical stability

    Returns:
        Normalized tensor of the same shape as x
    """
    if not _TRITON_AVAILABLE:
        return rms_norm_torch(x, weight, eps)

    batch, hidden_dim = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    _rms_norm_kernel[(batch,)](
        x,
        weight,
        out,
        x.stride(0),
        hidden_dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """PyTorch reference implementation (used for correctness checks and fixture generation)."""
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms * weight


def generate_fixture(batch: int, hidden_dim: int, seed: int = 42) -> Path:
    """Generate a deterministic test fixture and write it to tests/fixtures/.

    The fixture is a numpy .npz archive with keys:
        input   — shape [batch, hidden_dim], fp32
        weight  — shape [hidden_dim], fp32
        output  — shape [batch, hidden_dim], fp32  (PyTorch reference)

    Returns the path written.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn(batch, hidden_dim, generator=gen)
    weight = torch.randn(hidden_dim, generator=gen)
    output = rms_norm_torch(x, weight)

    fixtures_dir = Path(__file__).resolve().parent.parent / "cuda_oxide" / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    out_path = fixtures_dir / f"rms_norm_{batch}x{hidden_dim}.npz"
    np.savez(
        out_path,
        input=x.numpy(),
        weight=weight.numpy(),
        output=output.numpy(),
    )
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMS-Norm Triton baseline utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen_p = sub.add_parser("generate", help="Generate a correctness fixture (.npz)")
    gen_p.add_argument("--batch", type=int, default=4)
    gen_p.add_argument("--hidden_dim", type=int, default=4096)
    gen_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "generate":
        path = generate_fixture(args.batch, args.hidden_dim, args.seed)
        print(f"wrote {path}")
