# Phase 2 Plan: Triton Benchmark vs cuda-oxide on Narval A100

## Context

Phase 1 delivered a working cuda-oxide RMS-Norm kernel (scalar-naive, warp-shuffle reduction)
that passes correctness on Narval A100. Phase 2 benchmarks it against the already-complete
Triton baseline (`kernels/triton/rms_norm.py`) and a PyTorch reference across three shapes:
256×512 (small), 2048×4096 (Phase 1 baseline), and 4096×8192 (large). Results are logged to
MLflow under per-backend experiment names with p50, p99, and throughput (GB/s) metrics.
Grafana is deferred to Phase 3.

**Scope:**
- Triton kernel (`kernels/triton/rms_norm.py`) — complete as-is
- cuda-oxide kernel — bench mode added to `kernels/rms_norm_standalone/src/main.rs`
- No PyO3 bindings; cuda-oxide timed via subprocess from Python
- Single SLURM job, all three backends sequential on one A100
- Direct mlflow API calls (not `log_kernel_run()`) to log `throughput_gbs` correctly

---

## Files Changed

| Path | Action |
|---|---|
| `kernels/rms_norm_standalone/src/main.rs` | Modified: CLI args + `--bench` mode |
| `experiments/bench_rms_norm.py` | Created: Python benchmark harness |
| `kernels/triton/tests/test_rms_norm.py` | Created: Triton correctness tests |
| `slurm/bench_rms_norm_phase2.sh` | Created: Phase 2 SLURM job |
| `slurm/bench_job.sh` | Fixed: hardcoded username → `%u` / `$SCRATCH` |
| `kernels/CLAUDE.md` | Updated: rms_norm status → `complete` |
| `docs/plans/phase2-triton-benchmark.md` | Created: this document |

---

## Architecture

### cuda-oxide timing (`main.rs --bench`)

- Correctness check on `rms_norm_4x4096.npz` always runs first (gate)
- `--batch B --hidden-dim H` specifies bench shape (default 2048×4096)
- Warm up 100 iters, time 1000 iters; stream sync via 1-element D2H copy
- Prints single JSON line to stdout: `{"backend":"cuda_oxide","batch":...,"p50_us":...,...}`

### Python harness (`experiments/bench_rms_norm.py`)

- PyTorch / Triton: timed with `torch.cuda.Event`
- cuda_oxide: `subprocess.run([bin, "--batch", B, "--hidden-dim", H, "--bench"])`
- Logs per (backend, shape) to MLflow experiment `kernelserve/rms_norm/<backend>/narval/2026-05`
- Metrics: `latency_p50_us`, `latency_p99_us`, `throughput_gbs`

### MLflow experiment names (five-segment format per CLAUDE.md)

```
kernelserve/rms_norm/cuda_oxide/narval/2026-05
kernelserve/rms_norm/triton/narval/2026-05
kernelserve/rms_norm/pytorch/narval/2026-05
```

---

## Verification

```bash
# Local (no GPU):
pytest -m "not gpu"
uv run ruff check . && uv run mypy kernels experiments

# On Narval:
sbatch slurm/bench_rms_norm_phase2.sh
# Expect: 9 MLflow runs (3 backends × 3 shapes)
```

---

## Definition of Done

- [ ] `pytest -m "not gpu"` passes
- [ ] `cargo clippy --all-targets -- -D warnings` — zero warnings
- [ ] `uv run ruff check . && uv run mypy` clean
- [ ] SLURM job completes; 9 MLflow runs visible (3 backends × 3 shapes)
- [ ] All backends max abs error < 1e-4 vs PyTorch reference
- [ ] No Grafana changes (Phase 3)
