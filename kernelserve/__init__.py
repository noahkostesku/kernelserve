from __future__ import annotations

import torch

try:
    from kernelserve_core import rms_norm as _rms_norm_core
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False
    _rms_norm_core = None


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if not _CORE_AVAILABLE:
        raise RuntimeError(
            "kernelserve_core not built. Run: make build-bindings"
        )
    x_f32 = x.detach().cpu().to(torch.float32)
    w_f32 = weight.detach().cpu().to(torch.float32)
    out_flat = _rms_norm_core(x_f32.flatten().tolist(), w_f32.tolist(), eps)
    return torch.tensor(out_flat, dtype=torch.float32).reshape(x.shape)
