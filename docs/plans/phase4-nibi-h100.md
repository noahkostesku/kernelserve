# Phase 4 — Nibi H100 Benchmark Plan

## Goal

Re-run the Phase 3 RMS-Norm benchmark (3 backends × 3 shapes, OTel tracing, MLflow logging)
on the Nibi cluster's H100 80 GB GPUs and produce a direct A100 vs H100 comparison table.
No kernel or experiment code changes — SLURM script and infrastructure only.

---

## Hardware comparison

| | Narval (Phase 3) | Nibi (Phase 4) |
|---|---|---|
| GPU | A100 40 GB | H100 80 GB |
| SM target | sm_80 | sm_90 |
| SLURM gres | `a100:1` | `h100:1` |
| Memory directive | `--mem=32G` | `--mem-per-gpu=80G` |
| LLVM required | ≥ 18 | ≥ 21 |
| CUDA module | `cuda/12.9` | TODO — verify |
| LLVM module | `llvm/18.1.8` | TODO — verify |

---

## SLURM changes (phase3 → phase4)

| Directive / variable | Phase 3 (Narval) | Phase 4 (Nibi) |
|---|---|---|
| `--job-name` | `kernelserve-rms-norm-phase3` | `kernelserve-rms-norm-phase4-nibi` |
| `--gpus-per-node` | `a100:1` | `h100:1` |
| `--mem` | `--mem=32G` | `--mem-per-gpu=80G` |
| `CUDA_ARCH` | (unset, falls back to sm_80 in cargo args) | `export CUDA_ARCH=sm_90` |
| `CLUSTER` | (unset) | `export CLUSTER=nibi` |
| CUDA module | `cuda/12.9` | TODO |
| LLVM module | `llvm/18.1.8 clang/18.1.8` | TODO (llvmcore/21.1.5 ?) |
| cudacore cvmfs path | `cudacore/12.9.1/lib64` | TODO (depends on cuda version) |

Everything else — Python venv activation, correctness check, cargo oxide build, OTel env vars,
`srun python experiments/bench_rms_norm.py` — is identical to phase3.

---

## Open TODOs before submitting

### 1. CUDA version on Nibi
```bash
# ssh to Nibi login node, then:
module avail cuda
```
Phase 3 used `cuda/12.9` (Narval). Nibi likely has `cuda/12.6` or `cuda/12.9`; update the
`module load` line and the `cudacore` cvmfs path in `LD_LIBRARY_PATH` accordingly.

### 2. LLVM module on Nibi
```bash
module avail llvm
module avail llvmcore
```
sm_90 requires LLC ≥ 21. The script currently assumes `llvmcore/21.1.5` is available by
that module name on Nibi. If it differs, update both the `module load` line and `CUDA_OXIDE_LLC`.
The cvmfs path `/cvmfs/soft.computecanada.ca/.../llvmcore/21.1.5/bin/llc` may be the same
as Narval (shared cvmfs tree) — verify with `ls` on the Nibi login node.

### 3. Clang module on Nibi
`LIBCLANG_PATH` points to `clang/18.1.8`. Verify that module exists on Nibi:
```bash
module avail clang
```
If unavailable, check if `llvmcore/21.1.5` bundles clang and adjust `LIBCLANG_PATH` to
the llvmcore lib directory instead.

### 4. MLflow cluster tag in bench_rms_norm.py
`experiments/bench_rms_norm.py` builds the MLflow experiment name
`kernelserve/rms_norm/<backend>/nibi/<YYYY-MM>`. It must read the `CLUSTER` env var
(set to `nibi` in the phase4 SLURM script) rather than hardcoding `narval`. Verify
before submitting:
```bash
grep -n "CLUSTER\|narval\|nibi" experiments/bench_rms_norm.py
```
If `narval` is hardcoded, that is a one-line fix in `experiments/` (out of scope for this
scaffold PR; open a follow-up fix ticket).

### 5. Nibi SLURM partition
Narval uses the default partition. Nibi may require `--partition=gpu_h100` or similar.
Check Alliance docs or `sinfo` on the Nibi login node before submitting.

---

## MLflow experiment names

Phase 4 runs write to the five-segment format required by CLAUDE.md:

```
kernelserve/rms_norm/cuda_oxide/nibi/2026-05
kernelserve/rms_norm/triton/nibi/2026-05
kernelserve/rms_norm/pytorch/nibi/2026-05
```

Phase 3 Narval runs are in:
```
kernelserve/rms_norm/cuda_oxide/narval/2026-05
kernelserve/rms_norm/triton/narval/2026-05
kernelserve/rms_norm/pytorch/narval/2026-05
```

Both sets live under `file://$SCRATCH/mlruns`. The comparison query pulls both clusters
into one DataFrame by filtering on the `cluster` tag rather than experiment name.

---

## A100 vs H100 comparison

After both jobs complete, produce the comparison with:

```python
import mlflow, pandas as pd

client = mlflow.tracking.MlflowClient()

rows = []
for cluster in ("narval", "nibi"):
    for backend in ("cuda_oxide", "triton", "pytorch"):
        exp_name = f"kernelserve/rms_norm/{backend}/{cluster}/2026-05"
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            continue
        for run in client.search_runs(exp.experiment_id):
            rows.append({
                "cluster": cluster,
                "backend": backend,
                "shape": run.data.tags.get("shape"),
                "p50_us": run.data.metrics.get("latency_p50_us"),
                "p99_us": run.data.metrics.get("latency_p99_us"),
                "throughput_gbs": run.data.metrics.get("throughput_gbs"),
            })

df = pd.DataFrame(rows)
print(df.pivot_table(index=["backend", "shape"], columns="cluster",
                     values=["p50_us", "throughput_gbs"]))
```

Expected output: 9 rows per cluster (3 backends × 3 shapes), 18 rows total.

---

## Files created by this phase

| Path | Description |
|------|-------------|
| `slurm/bench_rms_norm_phase4_nibi.sh` | SLURM job for Nibi H100 |
| `docs/plans/phase4-nibi-h100.md` | This plan |

No kernel, experiment, serving, or observability code is changed in Phase 4.

---

## Verification

```bash
# 1. Static checks (no GPU required — no code changes so trivially passes)
pytest -m "not gpu"
uv run ruff check . && uv run mypy kernels serving observability experiments

# 2. Resolve all TODOs above by running module avail on Nibi login node

# 3. Dry-run (check SLURM script parses cleanly)
sbatch --test-only slurm/bench_rms_norm_phase4_nibi.sh

# 4. Submit to Nibi
sbatch slurm/bench_rms_norm_phase4_nibi.sh

# 5. Confirm outputs
#    - seff <JOBID> — GPU utilization > 0
#    - $SCRATCH/traces/bench_<JOBID>.jsonl — 9 span lines, sm.target=sm_90
#    - MLflow — 9 runs under kernelserve/rms_norm/*/nibi/2026-05
```

## Definition of Done

- [ ] All four TODOs above resolved (CUDA version, LLVM module, clang, partition)
- [ ] `sbatch --test-only` exits 0
- [ ] SLURM job completes without error on Nibi
- [ ] `seff <JOBID>` shows GPU utilization > 0
- [ ] 9 MLflow runs logged under `nibi/2026-05` experiments
- [ ] Span file contains 9 entries with `sm.target=sm_90`
- [ ] A100 vs H100 comparison DataFrame renders (18 rows, 2 clusters)
