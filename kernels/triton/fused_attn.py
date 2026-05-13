# Triton baseline for fused QKV attention.
#
# This is the reference implementation used for:
# 1. Correctness validation against kernels/cuda_oxide/src/fused_attn.rs
# 2. Performance comparison in tests/benchmark/
#
# PyTorch reference: fused_attn_torch() below
# Rust counterpart:  kernels/cuda_oxide/src/fused_attn.rs

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attn_kernel(
    qkv_ptr,
    out_ptr,
    stride_b,
    stride_s,
    seq_len,
    num_heads,
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # TODO: implement Triton fused attention kernel (FlashAttention-style tiling)
    # 1. pid_b = program_id(0), pid_h = program_id(1), pid_m = program_id(2)
    # 2. Load Q tile from qkv_ptr (first head_dim cols)
    # 3. Loop over K/V tiles, compute softmax(QK^T * scale) online (numerically stable)
    # 4. Accumulate weighted V
    # 5. Store output tile
    pass


def fused_attn_triton(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Launch the Triton fused attention kernel.

    Args:
        qkv:       Packed tensor of shape [batch, seq_len, 3 * num_heads * head_dim], fp32, CUDA
        num_heads: Number of attention heads
        head_dim:  Dimension per head (must be power of 2, <= 128 for sm_80)

    Returns:
        Attention output of shape [batch, seq_len, num_heads * head_dim]
    """
    # TODO: validate shapes, compute grid, launch _fused_attn_kernel
    raise NotImplementedError("fused_attn_triton not yet implemented")


def fused_attn_torch(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """PyTorch reference implementation (used for correctness checks)."""
    batch, seq_len, _ = qkv.shape
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v = qkv.split(num_heads * head_dim, dim=-1)
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)
