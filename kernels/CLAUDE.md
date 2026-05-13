# CLAUDE.md — kernels/

## What lives here

- `cuda_oxide/` — Rust GPU kernels using cuda-oxide (Phase 1: sm_80 A100)
- `triton/` — Python Triton kernel baselines (same op, used for correctness comparison)

## Kernel registry

| Name | File | sm target | Status |
|---|---|---|---|
| rms_norm | `cuda_oxide/src/rms_norm.rs` | sm_80 | stub |
| fused_attn | `cuda_oxide/src/fused_attn.rs` | sm_80 | stub |

Update this table when adding or completing a kernel.

## Rules

- Every kernel in `cuda_oxide/` must have a matching Triton baseline in `triton/`
- Correctness threshold: max abs error vs PyTorch reference < 1e-4 (fp32) or < 1e-2 (fp16)
- No sm_90 intrinsics until Phase 2 is declared in the root `CLAUDE.md`
- Rust: no `unwrap()` in `src/` — only in `tests/`
- Keep each `.rs` kernel file under 300 lines; extract helpers to a `utils` module if needed

## Build

```bash
cd kernels/cuda_oxide
module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2/lib64:${LD_LIBRARY_PATH}
cargo build --release
cargo test
```
