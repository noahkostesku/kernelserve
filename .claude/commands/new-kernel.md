---
description: Scaffold a new kernel with stub files in cuda_oxide/src/, kernels/triton/, and tests/benchmark/
argument-hint: <kernel-name>
allowed-tools: [Bash, Read, Write, Edit]
---

# /new-kernel — Kernel Scaffolder

Kernel name: $ARGUMENTS

## What this command creates

1. `kernels/cuda_oxide/src/<kernel_name>.rs` — Rust stub with `pub struct` and `todo!()` forward pass
2. `kernels/triton/<kernel_name>.py` — Triton stub with `@triton.jit` decorator and PyTorch reference
3. Adds `pub mod <kernel_name>;` and `pub use <kernel_name>::*;` to `kernels/cuda_oxide/src/lib.rs`
4. `tests/benchmark/test_<kernel_name>.py` — pytest-benchmark stub for all three impls

## Steps

1. Validate that $ARGUMENTS is a valid Rust identifier (snake_case, no hyphens)
2. Check that the name doesn't already exist in `kernels/cuda_oxide/src/`
3. Create the three files listed above
4. Run `cargo check` in `kernels/cuda_oxide/` to verify the new module compiles
5. Run `pytest tests/benchmark/test_<kernel_name>.py --collect-only` to verify test discovery
6. Update the kernel registry table in `kernels/CLAUDE.md`
7. Report the created file paths and remind the user to implement the TODOs

## Naming conventions

- Use snake_case: `rms_norm`, `fused_attn`, `flash_attn_v2`
- Do not prefix with `cuda_` or `triton_` — the directory provides that context
