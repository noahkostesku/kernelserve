from __future__ import annotations

import torch

from kernelserve_core import rms_norm as _rms_norm_core


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """RMS-Norm via cuda-oxide (GPU) or CPU reference (KERNELSERVE_DEVICE=cpu).

    Args:
        x:      Input tensor of shape [batch, hidden_dim], float32.
        weight: Scale parameter of shape [hidden_dim], float32.
        eps:    Numerical stability epsilon.

    Returns:
        Normalized tensor of the same shape as x.
    """
    x_f32 = x.detach().cpu().to(torch.float32)
    w_f32 = weight.detach().cpu().to(torch.float32)
    out_flat = _rms_norm_core(x_f32.flatten().tolist(), w_f32.tolist(), eps)
    return torch.tensor(out_flat, dtype=torch.float32).reshape(x.shape)
