# Phase 1 Implementation Plan: RMS-Norm Kernel on Narval A100

## Context

Phase 1 goal: build cuda-oxide on the Alliance Canada Narval cluster (A100 sm_80) and deliver a
working scalar-naive RMS-Norm kernel that passes numeric correctness tests against PyTorch.

**Parameters locked in:**
- Cluster: Narval, `CUDA_ARCH=sm_80`, account `def-cbravo`
- LLVM: `llvm/18.1.8` (satisfies sm_80 ≥ 18; llvm/21 deferred to Phase 2 / sm_90)
- Benchmark shape: 2048 × 4096 (LLaMA-2 7B hidden dim)
- Kernel style: scalar naive first, warp-tiled as a follow-on optimisation

---

## Step 0 — Prerequisite: pin cuda-oxide upstream

**Blocker.** `kernels/cuda_oxide/Cargo.toml` has a placeholder dep:
```toml
# cuda-oxide = { git = "https://github.com/<org>/cuda-oxide", rev = "<commit-sha>" }
```

1. Confirm the upstream repo URL and a stable commit SHA.
2. Replace the placeholder:
   ```toml
   cuda-oxide = { git = "https://github.com/<confirmed-org>/cuda-oxide", rev = "<sha>" }
   ```
3. `cargo fetch` — verify Cargo resolves the dep (no GPU needed).
4. `cargo check` — confirm the API surface compiles cleanly.

Do not proceed until `cargo check` is green.

---

## Step 1 — Fix bugs in slurm/bench_job.sh  *(already applied)*

| Bug | Fix |
|---|---|
| `cargo build --release` skips PTX compilation | Changed to `cargo oxide build --release` |
| venv path `$SCRATCH/kernelserve-venv` mismatched `setup_env.sh` | Changed to `$REPO_DIR/.venv` |
| `llvm/18.1.8` missing from module load | Added to the `module load` line |

---

## Step 2 — Environment setup on Narval (exact module loads)

`slurm/scripts/setup_env.sh` already contains the correct module names. Source it, then verify:

```bash
source slurm/scripts/setup_env.sh

rustc --version       # expect 1.91.0
nvcc --version        # expect 12.2.x
llvm-config --version # expect 18.1.8
```

Create the venv once:
```bash
python -m venv .venv
source .venv/bin/activate
uv sync
```

---

## Step 3 — Implement the scalar-naive RMS-Norm kernel

**File:** `kernels/cuda_oxide/src/rms_norm.rs`

The struct, error enum, and `forward` signature are already correct — implement the kernel body only.

### Formula
```
y_i = x_i / sqrt( (1/H) * Σ x_j² + ε ) * w_i
```

### Launch config (2048 × 4096)
| Parameter | Value | Reasoning |
|---|---|---|
| Grid | `(batch_size,)` = `(2048,)` | one block per row |
| Block | `(128,)` threads | 4 warps, fits 4096-wide row |
| Elements/thread | `4096 / 128 = 32` | sequential load stride |

### Kernel passes (within each block)
1. Each thread accumulates `sum_sq` over its 32 elements.
2. Warp shuffle reduction (`__shfl_down_sync`) reduces 128 threads → 4 partial sums in shared mem.
3. Thread 0 computes `rms = sqrt(total_sum_sq / hidden_dim + eps)`.
4. Broadcast `rms` via shared memory (syncthreads).
5. Each thread writes its 32 normalized+scaled outputs.

### cuda-oxide rules
- `#[kernel]` attribute on the device function
- `thread::index()` / `block::index()` for indexing
- No `unwrap()` / `expect()` in `src/` — use `?` or `map_err`
- Every `unsafe` block needs a `// SAFETY:` comment

---

## Step 4 — Triton baseline

**File:** `kernels/triton/rms_norm.py`

Must contain:
- `rms_norm_triton(x, weight, eps)` — `@triton.jit` kernel
- `rms_norm_torch(x, weight, eps)` — CPU reference for fixture generation
- `generate_fixture(batch, hidden_dim, seed)` — writes
  `kernels/cuda_oxide/tests/fixtures/rms_norm_<batch>x<hidden_dim>.npz`
  with keys `input`, `weight`, `output` (all fp32)

Generate fixtures locally (CPU mock mode works on macOS):
```bash
python kernels/triton/rms_norm.py generate --batch 4 --hidden_dim 4096 --seed 42
```

---

## Step 5 — Correctness test

**File:** `kernels/cuda_oxide/tests/correctness.rs`

Replace the `test_rms_norm_vs_pytorch` placeholder:
1. Load `tests/fixtures/rms_norm_4x4096.npz`
2. Call `RmsNorm::new(1e-5).forward(&input, &weight)?`
3. Assert element-wise: `assert_abs_diff_eq!(result[i], reference[i], epsilon = 1e-4)`

Keep the `#[ignore = "requires A100 GPU"]` marker.

Run on Narval:
```bash
cd kernels/cuda_oxide
cargo test --test correctness -- --ignored
```

**Pass criterion:** max absolute error < 1e-4 across all 4 × 4096 = 16,384 elements.

---

## Step 6 — Benchmark job

Submit:
```bash
KERNEL_NAME=rms_norm sbatch slurm/bench_job.sh
```

**MLflow experiment name:**
```
kernelserve/rms_norm/cuda_oxide/narval/2026-05
```

**SLURM time budget** (`--time=0-01:00` in bench_job.sh provides margin):
| Phase | Estimate |
|---|---|
| `cargo oxide build --release` | ~10 min |
| Correctness test | ~2 min |
| Benchmark (100 warm-up + 1000 timed iters) | ~15 min |
| Total | ~27 min |

---

## Definition of Done

- [ ] `cargo check` green with pinned cuda-oxide dep
- [ ] `cargo clippy --all-targets -- -D warnings` — zero warnings
- [ ] `pytest -m "not gpu"` passes
- [ ] `cargo test --test correctness -- --ignored` passes on Narval (max abs error < 1e-4)
- [ ] SLURM job completes; MLflow run visible under `kernelserve/rms_norm/cuda_oxide/narval/2026-05`
- [ ] `kernels/CLAUDE.md` rms_norm row updated to `complete`
- [ ] Changes on branch `feat/rms-norm-phase1`, not committed directly to `main`
