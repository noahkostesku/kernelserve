from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kernels.triton.rms_norm import rms_norm_torch, rms_norm_triton

FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "kernels"
    / "cuda_oxide"
    / "tests"
    / "fixtures"
    / "rms_norm_4x4096.npz"
)
ATOL = 1e-4


def test_rms_norm_torch_vs_fixture() -> None:
    if not FIXTURE.exists():
        pytest.skip(f"fixture not found at {FIXTURE}")
    data = np.load(FIXTURE)
    x = torch.from_numpy(data["input"])
    w = torch.from_numpy(data["weight"])
    ref = torch.from_numpy(data["output"])
    out = rms_norm_torch(x, w)
    max_err = float((out - ref).abs().max())
    assert max_err < ATOL, f"max abs error {max_err:.2e} ≥ threshold {ATOL:.1e}"


@pytest.mark.gpu
def test_rms_norm_triton_vs_torch() -> None:
    torch.manual_seed(42)
    x = torch.randn(4, 4096, device="cuda", dtype=torch.float32)
    w = torch.randn(4096, device="cuda", dtype=torch.float32)
    ref = rms_norm_torch(x, w)
    out = rms_norm_triton(x, w)
    max_err = float((out.cpu() - ref.cpu()).abs().max())
    assert max_err < ATOL, f"max abs error {max_err:.2e} ≥ threshold {ATOL:.1e}"
