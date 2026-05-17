# KernelServe

Benchmarks Rust/cuda-oxide GPU kernels against Triton and PyTorch baselines.
Served via NVIDIA Triton Inference Server + Ray Serve. Tracked with MLflow.
Compute: Alliance Canada HPC (default: Narval A100 40 GB sm_80). Each layer has its own `CLAUDE.md` — read it before working there.

## Commands

```bash
make test                                                    # before every Python push (build-bindings + pytest -m "not gpu")
cargo clippy --all-targets -- -D warnings                    # before every Rust push; zero warnings required
uv run ruff check . && uv run mypy kernels serving observability experiments kernelserve
cd kernels/cuda_oxide && cargo oxide build --release         # NOT cargo build
sbatch slurm/bench_job.sh                                    # submit benchmark; use srun inside sbatch, never raw python

# Phase 5 — local usability
make dev                                                     # full bootstrap: uv sync + build-bindings
make build-bindings                                          # rebuild CPU PyO3 binding into .venv
make build-bindings-gpu                                      # rebuild GPU binding (cargo oxide first)
make test                                                    # build-bindings then pytest -m "not gpu"
uv run ks bench --kernel rms_norm                           # benchmark (JSON + optional MLflow)
uv run ks compare --kernel rms_norm                         # side-by-side table
docker-compose --profile kernelserve up kernelserve         # CPU mock container (no nvidia runtime)
```

## Layout

| Layer | Directory | Language |
|---|---|---|
| Custom kernels | `kernels/cuda_oxide/` | Rust (cuda-oxide) |
| PyO3 bindings | `kernels/pyo3_bindings/` | Rust (maturin/PyO3) |
| Baseline kernels | `kernels/triton/` | Python (Triton) |
| Python package + CLI | `kernelserve/` | Python |
| Serving | `serving/` | Python |
| Observability | `observability/` | Python / YAML |
| Experiments | `experiments/` | Python |
| HPC jobs | `slurm/` | Bash |

## Tech stack constraints

- GPU kernels: Rust + cuda-oxide only — no raw CUDA C
- Python: 3.11+; use `uv` for all package management
- Serving runtime: Triton Inference Server only — no vLLM, no TGI
- Orchestration: Ray Serve in local mode — no Kubernetes, no `ray.init(address="auto")`
- Experiment tracking: MLflow only — no W&B, no custom loggers
- Tracing: OpenTelemetry only — no Datadog SDK

## Architecture rules

- Every cuda-oxide kernel needs a matching Triton baseline in `kernels/triton/`
- All SLURM scripts live in `slurm/` — nowhere else
- `docker-compose.yml` is for local serving stack testing only — not a deployment target
- Read GPU counts and node names from SLURM env vars; never hardcode them

## MLflow

- Experiment name format (all five segments required): `kernelserve/<kernel>/<backend>/<cluster>/<YYYY-MM>`
  - Example: `kernelserve/rms_norm/cuda_oxide/narval/2026-05`
  - Local runs: use `local` as `<cluster>` — e.g. `kernelserve/rms_norm/cpu_ref/local/2026-05`
  - Short names like `kernelserve/rms_norm` mix Phase 1 and Phase 2 runs — reject them
- Tag every run: `mlflow.set_tag("git_sha", subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode())`
- MLflow URI on HPC: `file://$SCRATCH/mlruns`; locally defaults to `file://./mlruns`

## HPC

- Default cluster: Narval (A100 40 GB, sm_80); Phase 2 only: Nibi (H100 80 GB, sm_90)
- Read GPU arch from env — never hardcode `sm_80` or `sm_90` in source files
- Phase 2 not active; write no sm_90 / Hopper (TMA, wgmma) code until Phase 2 is declared
- Account: `def-cbravo`
- Module loads: `module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0 python/3.11`
- LLVM: sm_80 requires ≥ 18; sm_90 (Phase 2) requires ≥ 21
- `triton` wheels are Linux+CUDA only — macOS runs CPU mock mode

## Git rules

- Branch prefixes: `feat/`, `fix/`, `bench/`, `obs/`
- Never commit directly to `main`
- Commit messages: imperative mood, lowercase, under 72 chars
- Run `pytest -m "not gpu"` and linters before every commit
- Never commit `.env`, `.env.local`, raw `.csv`, or `.out` benchmark files

## Subagent delegation

- Kernel work (`kernels/`) → `kernel-dev` subagent with `isolation: worktree`
- Serving work (`serving/`) → `serving-dev` subagent with `isolation: worktree`
- Any benchmark run → `benchmarker` subagent

## What Claude gets wrong here

- Build command is `cargo oxide build`, not `cargo build` — `cargo build` silently skips PTX compilation
- Triton backend `config.pbtxt` uses `backend: "python"`, not `backend: "pytorch"`
- Ray Serve runs in local mode on HPC; calling `ray.init(address="auto")` hangs waiting for a cluster head node
- `unwrap()` and `expect()` are banned in `kernels/cuda_oxide/src/`; allowed only in `tests/`
- Correctness threshold is max abs error < 1e-4 (fp32) vs PyTorch reference — "looks close" is not a passing test
- After changing a Triton kernel signature, bump `version` or clear `~/.triton/` cache; stale cache silently runs old code
- PyO3 GPU build is a 2-step process: `cargo oxide build` first (embeds PTX), then `maturin develop --features gpu`. Maturin alone cannot compile cuda-oxide kernels.
- `kernels/pyo3_bindings/` module-name `kernelserve._core` requires the Rust `#[pymodule]` function to be named `_core` (last segment of module path)
- `KERNELSERVE_DEVICE=cpu` forces CPU reference path regardless of GPU availability
