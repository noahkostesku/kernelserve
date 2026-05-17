//! kernelserve-kernels — Custom CUDA GPU kernels (Phase 1: sm_80 A100)
//!
//! Each module contains one kernel implementation. All kernels must have
//! a matching Triton baseline in `kernels/triton/` and a correctness test
//! in `tests/correctness.rs`.

pub mod cpu_ref;
pub mod fused_attn;
pub mod rms_norm;

pub use fused_attn::FusedAttn;
pub use rms_norm::RmsNorm;
