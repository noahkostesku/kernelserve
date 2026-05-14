# kernels/CLAUDE.md

| Kernel | File | Status |
|---|---|---|
| rms_norm | `cuda_oxide/src/rms_norm.rs` | stub |
| fused_attn | `cuda_oxide/src/fused_attn.rs` | stub |

Update this table when adding or completing a kernel.

## Rules

- Build: `cargo oxide build` — not `cargo build`, which silently skips PTX compilation
- PTX output goes to `kernels/cuda_oxide/ptx/` (gitignored; `.gitkeep` is committed)
- Read GPU arch from `$CUDA_ARCH`; default sm_80 — never hardcode arch in Cargo.toml or build scripts
- One kernel per `.rs` file, named after the operation; keep each file under 300 lines
- Every kernel needs a correctness test in `tests/unit/` vs PyTorch reference (`atol=1e-3, rtol=1e-3`)
- Every kernel needs a Python binding via PyO3 or a Triton custom backend
- `unsafe` blocks require a `// SAFETY:` comment explaining the invariant
- Run `/bench` before and after any kernel change
