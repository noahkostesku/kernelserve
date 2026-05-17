//! CPU reference implementation of RMS-Norm.
//!
//! Used by the PyO3 bindings when built without --features gpu,
//! and by correctness tests on machines without a CUDA device.
//!
//! Formula: y[i] = x[i] / sqrt(mean(x²) + eps) * weight[i]

/// Compute RMS-Norm on the CPU.
///
/// # Arguments
/// * `input`      — row-major f32 tensor, length `batch * hidden`
/// * `weight`     — scale parameter, length `hidden`
/// * `batch`      — number of rows
/// * `hidden_dim` — number of columns / features per row
/// * `eps`        — numerical stability epsilon (typically 1e-5)
pub fn rms_norm_cpu(
    input: &[f32],
    weight: &[f32],
    batch: usize,
    hidden_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * hidden_dim];
    for row in 0..batch {
        let base = row * hidden_dim;
        let sum_sq: f32 = input[base..base + hidden_dim].iter().map(|&x| x * x).sum();
        let rms_inv = 1.0_f32 / (sum_sq / hidden_dim as f32 + eps).sqrt();
        for col in 0..hidden_dim {
            out[base + col] = input[base + col] * rms_inv * weight[col];
        }
    }
    out
}
