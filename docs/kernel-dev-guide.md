# Kernel Development Guide

## Overview

Kernels live in two forms:
- **`kernels/cuda_oxide/`** — Rust implementation (the primary artifact)
- **`kernels/triton/`** — Python/Triton baseline (for correctness + comparison)

Both implementations must produce numerically equivalent results (max abs error < 1e-4 fp32).

## Setting up cuda-oxide

cuda-oxide is a Rust crate for writing CUDA kernels. It requires:

- CUDA 12.x headers and `libcuda.so` on the library path
- Rust stable (≥ 1.75)
- LLVM ≥ 18 (for sm_80/A100, Phase 1); LLVM ≥ 21 for sm_90/H100 (Phase 2)

**On Narval:**
```bash
module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2/lib64:${LD_LIBRARY_PATH}
```

**Cargo dependency** (fill in after upstream is confirmed):
```toml
# kernels/cuda_oxide/Cargo.toml
cuda-oxide = { git = "https://github.com/<org>/cuda-oxide", rev = "<commit-sha>" }
```

## PTX pipeline

cuda-oxide compiles Rust to PTX (Parallel Thread eXecution) assembly, which is then JIT-compiled
by the CUDA driver for the target architecture at runtime.

```
Rust source (.rs)
    → cuda-oxide → NVVM IR
    → LLVM NVPTX backend → PTX (.ptx)
    → CUDA driver → SASS (sm_80 binary)
```

Key things to understand:
- `#[kernel]` attribute marks a function as a CUDA kernel
- Shared memory is declared with `__shared__` equivalents in cuda-oxide
- Thread/block indexing uses `thread::index()` and `block::index()` intrinsics

## Writing a new kernel

1. Use `/new-kernel <name>` to scaffold the files
2. Implement in `kernels/cuda_oxide/src/<name>.rs` — start with a naive implementation
3. Implement the Triton baseline in `kernels/triton/<name>.py`
4. Add a PyTorch CPU reference in the same Triton file for correctness checking
5. Write a correctness test in `kernels/cuda_oxide/tests/correctness.rs`
6. Run benchmarks: `/bench <name>`

## sm_80 (A100) constraints and tips

- Shared memory per SM: 164 KB (Phase 1 target)
- Max threads per block: 1024
- Max blocks per SM: 32
- Warp size: 32 threads
- For attention kernels: seq_len ≤ 2048 fits in shared memory with head_dim ≤ 128
- Prefer 128-byte aligned memory accesses for coalescing

## Common gotchas

- **No `unwrap()` in `src/`** — use `?` propagation or map to a typed error
- **Always run `cargo clippy -- -D warnings`** before pushing — CI enforces this
- **Triton kernel cache** — Triton caches compiled PTX in `~/.triton/`. If you change the kernel
  signature, clear the cache or change the `version` parameter to force recompilation
- **sm_90 intrinsics are off-limits in Phase 1** — `wgmma`, TMA tile loads are H100-only

## Running correctness tests

```bash
cd kernels/cuda_oxide
# Skip on CI (no GPU), run manually on Narval
cargo test --test correctness -- --ignored
```
