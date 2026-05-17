# Phase 5: KernelServe Local GPU Usability

## Problem

KernelServe was built for Alliance Canada HPC (Narval). Any researcher with a
local GPU — or a CPU-only laptop — could not run it without a SLURM cluster
account, `module load` access, and the right environment variables. Phase 5
removes that dependency.

## Goals

1. `import kernelserve; kernelserve.rms_norm(x, w)` works on any machine
2. `ks bench` and `ks compare` CLI commands work without SLURM
3. `docker-compose --profile kernelserve up` works without an nvidia runtime

Out of scope: Triton IS backend (`serving/triton_backends/`), Ray Serve
(`serving/ray_serve/`), and new kernels.

## Target user

Researcher who writes a kernel → builds → benches → logs to MLflow.

---

## Architecture

```
kernelserve.rms_norm(x: Tensor, w: Tensor) -> Tensor   # kernelserve/__init__.py
        │
        ▼
kernelserve._core.rms_norm(flat_list, weight_list, eps)  # PyO3 .so
        │
        ├── [feature = "gpu"]   → RmsNorm::forward()     # cuda-oxide GPU kernel
        └── [no gpu feature]    → cpu_ref::rms_norm_cpu() # pure Rust CPU reference
```

The PyO3 binding lives in `kernels/pyo3_bindings/` — a separate `cdylib` crate.
It is NOT inside `kernels/cuda_oxide/` because the cuda_oxide crate uses
`cargo oxide build` proc-macros that maturin cannot drive.

---

## Key files

| File | Purpose |
|---|---|
| `kernels/cuda_oxide/src/cpu_ref.rs` | CPU reference `rms_norm_cpu()` |
| `kernels/pyo3_bindings/Cargo.toml` | cdylib crate; pyo3 dep |
| `kernels/pyo3_bindings/pyproject.toml` | maturin config; module-name `kernelserve._core` |
| `kernels/pyo3_bindings/src/lib.rs` | `#[pymodule]` + `#[pyfunction] rms_norm()` |
| `kernelserve/__init__.py` | Torch tensor wrapper |
| `kernelserve/cli/main.py` | `ks` entry point |
| `kernelserve/cli/bench.py` | `ks bench` — timing loop + MLflow |
| `kernelserve/cli/compare.py` | `ks compare` — 3-backend table |
| `tests/unit/test_pyo3_cpu.py` | CPU correctness gate |
| `Dockerfile.dev` | CPU-only dev image |

---

## Build workflows

### CPU-only (macOS, no CUDA)

```bash
cd kernels/pyo3_bindings
maturin develop         # compiles cpu_ref path, installs kernelserve._core
uv run ks bench --kernel rms_norm
```

### GPU (Linux + CUDA)

```bash
# Step 1: compile PTX via cuda-oxide toolchain
cd kernels/cuda_oxide
cargo oxide build --release --features gpu

# Step 2: link and install the PyO3 extension
cd ../pyo3_bindings
maturin develop --features gpu

uv run ks bench --kernel rms_norm --log-mlflow
```

Maturin alone cannot compile cuda-oxide kernels — step 1 is mandatory for GPU builds.

---

## MLflow experiment naming

Local runs use `local` as the cluster segment:

```
kernelserve/rms_norm/cpu_ref/local/2026-05
kernelserve/rms_norm/cuda_oxide/local/2026-05
```

Cluster is auto-detected: `SLURM_JOB_ID` present → `narval`/`nibi`; absent → `local`.
Override with `CLUSTER=<name>` env var.

---

## Docker

```bash
# Start CPU mock container (no nvidia runtime, no GPU required)
docker-compose --profile kernelserve up kernelserve

# Base `docker-compose up` (no profile) still starts only observability stack
docker-compose up
```

The `kernelserve` service is opt-in via a Docker Compose profile so existing
users of the observability stack are not affected.

---

## Correctness gate

`tests/unit/test_pyo3_cpu.py` runs the cpu_ref path against the PyTorch-generated
fixture at `kernels/cuda_oxide/tests/fixtures/rms_norm_4x4096.npz`.

Threshold: max absolute error < 1e-4 (same as the GPU kernel).

Run without a GPU:

```bash
cd kernels/pyo3_bindings && maturin develop
pytest tests/unit/test_pyo3_cpu.py -v
```
