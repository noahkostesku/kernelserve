# CLAUDE.md — KernelServe

AI tooling guide. For human onboarding see `docs/architecture.md`.

## What this project is

KernelServe benchmarks custom GPU kernels written in Rust (cuda-oxide) against Triton and
PyTorch baselines, serves them through NVIDIA Triton Inference Server and Ray Serve, and
tracks experiments with MLflow. Compute target: Alliance Canada HPC (Narval A100 sm_80;
Phase 2 Nibi H100 sm_90).

## Repo layout

| Layer | Directory | Language |
|---|---|---|
| Custom kernels | `kernels/cuda_oxide/` | Rust (cuda-oxide) |
| Baseline kernels | `kernels/triton/` | Python (Triton) |
| Serving | `serving/` | Python |
| Observability | `observability/` | Python / YAML |
| Experiments | `experiments/` | Python |
| HPC jobs | `slurm/` | Bash |

Each layer has its own `CLAUDE.md` — read it before working in that layer.

## Cross-cutting rules

- Phase 1 target: Narval (A100 40 GB, sm_80). Never hard-code sm_90 paths in Phase 1 code.
- All benchmark results go in MLflow. Do not commit raw `.csv` or `.out` result files.
- Never commit `.env` or `.env.local` — use `.env.example` stubs only.
- SLURM scripts live in `slurm/` only. Never scatter job scripts elsewhere.
- Python: use `ruff` for formatting/linting, `mypy` for types.
- Rust: `clippy -D warnings` is CI-enforced. No `unwrap()` in library code (`src/`).

## Quick commands

```bash
# Run Python tests (no GPU required)
pytest -m "not gpu"

# Run correctness tests for CUDA kernels (requires A100)
pytest tests/ -m gpu

# Submit benchmark job to Narval
sbatch slurm/bench_job.sh

# Launch Ray Serve locally
python serving/ray_serve/deployment.py
```

## Agents

- `/kernel-dev` — writing and testing CUDA kernels
- `/serving-dev` — Triton backend and Ray Serve configs
- `/benchmarker` — running and interpreting benchmark results

## Known constraints

- `triton` wheels only ship for Linux + CUDA. macOS dev uses CPU-only mock mode.
- `cuda-oxide` requires CUDA 12.x headers. On Narval: `module load StdEnv/2023 cuda/12.2`.
- `ray[serve]` pins `starlette`; do not upgrade starlette independently.
- sm_80 (A100) works with LLVM 18+. sm_90 (H100, Phase 2) requires LLVM 21+.
