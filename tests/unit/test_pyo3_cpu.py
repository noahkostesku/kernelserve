from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import torch

_SKIP = pytest.mark.skipif(
    importlib.util.find_spec("kernelserve_core") is None,
    reason="kernelserve_core not built — run make build-bindings first",
)

ATOL = 1e-4


def _rms_norm_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    return x / rms * w


@_SKIP
def test_rms_norm_output_shape() -> None:
    import kernelserve_core

    x = np.random.randn(4, 512).astype(np.float32)
    w = np.ones(512, dtype=np.float32)
    out = kernelserve_core.rms_norm(x, w)
    assert out.shape == x.shape


@_SKIP
def test_rms_norm_matches_torch_small() -> None:
    import kernelserve_core

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((4, 512)).astype(np.float32)
    w_np = rng.standard_normal(512).astype(np.float32)

    out_np = kernelserve_core.rms_norm(x_np, w_np)

    x_t = torch.from_numpy(x_np)
    w_t = torch.from_numpy(w_np)
    ref = _rms_norm_ref(x_t, w_t).numpy()

    max_err = float(np.abs(out_np - ref).max())
    assert max_err < ATOL, f"max abs error {max_err:.2e} ≥ threshold {ATOL:.1e}"


@_SKIP
def test_rms_norm_matches_torch_large() -> None:
    import kernelserve_core

    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2048, 4096)).astype(np.float32)
    w_np = rng.standard_normal(4096).astype(np.float32)

    out_np = kernelserve_core.rms_norm(x_np, w_np)

    x_t = torch.from_numpy(x_np)
    w_t = torch.from_numpy(w_np)
    ref = _rms_norm_ref(x_t, w_t).numpy()

    max_err = float(np.abs(out_np - ref).max())
    assert max_err < ATOL, f"max abs error {max_err:.2e} ≥ threshold {ATOL:.1e}"


@_SKIP
def test_rms_norm_unit_weight_is_normalize() -> None:
    import kernelserve_core

    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((8, 256)).astype(np.float32)
    w_np = np.ones(256, dtype=np.float32)

    out_np = kernelserve_core.rms_norm(x_np, w_np)

    row_rms = np.sqrt((out_np**2).mean(axis=-1))
    assert np.allclose(row_rms, 1.0, atol=ATOL), f"rows not unit RMS: {row_rms}"
