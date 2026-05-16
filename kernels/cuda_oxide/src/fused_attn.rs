//! Fused attention kernel (sm_80, A100).
//!
//! Implements fused QKV projection + scaled dot-product attention in a single kernel,
//! avoiding intermediate materialization of Q, K, V tensors.
//!
//! References:
//! - FlashAttention: <https://arxiv.org/abs/2205.14135>
//! - Triton baseline: `kernels/triton/fused_attn.py`

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FusedAttnError {
    #[error("sequence length {0} exceeds maximum supported {1}")]
    SeqLenExceeded(usize, usize),
    #[error("head_dim must be a power of 2, got {0}")]
    InvalidHeadDim(usize),
    #[error("CUDA kernel launch failed: {0}")]
    CudaLaunch(String),
}

/// Fused QKV attention kernel wrapper (sm_80).
pub struct FusedAttn {
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl FusedAttn {
    /// Create a new FusedAttn kernel.
    ///
    /// * `num_heads` — number of attention heads
    /// * `head_dim`  — dimension per head (must be power of 2, ≤ 128 for sm_80)
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            num_heads,
            head_dim,
            scale,
        }
    }

    /// Forward pass: compute attention output from packed QKV input.
    ///
    /// # Arguments
    /// * `qkv` — packed tensor of shape `[batch, seq_len, 3 * num_heads * head_dim]`
    ///
    /// # Errors
    ///
    /// Returns [`FusedAttnError::SeqLenExceeded`] if seq_len > 2048 (sm_80 shared memory limit).
    /// Returns [`FusedAttnError::InvalidHeadDim`] if head_dim is not a power of 2.
    pub fn forward(&self, _qkv: &[f32]) -> Result<Vec<f32>, FusedAttnError> {
        // TODO: implement fused attention cuda-oxide kernel
        // 1. Validate seq_len <= 2048 (tiling constraint for sm_80 SMEM budget: 164 KB)
        // 2. Validate head_dim is power of 2
        // 3. Allocate device buffers: qkv_dev, output_dev
        // 4. Copy host → device
        // 5. Launch fused_attn_kernel<<<grid, block, smem>>>(qkv, output, scale, seq_len, head_dim)
        // 6. Copy output device → host
        let _ = (self.num_heads, self.head_dim, self.scale);
        todo!("fused_attn forward pass not yet implemented")
    }
}
