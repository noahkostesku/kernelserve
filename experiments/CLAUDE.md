# CLAUDE.md — experiments/

## What lives here

- `mlflow_setup.py` — MLflow tracking server setup and run logging helpers
- `runs/` — placeholder directory (actual MLflow artifacts live in `$SCRATCH/mlruns/` on Narval)

## Rules

- Every benchmark run must be tagged with the git commit SHA: `mlflow.set_tag("git_sha", ...)`
- Phase 1 (Narval A100 sm_80) and Phase 2 (Nibi H100 sm_90) results use **separate** experiment names:
  - Phase 1: `kernelserve/narval/<kernel_name>`
  - Phase 2: `kernelserve/nibi/<kernel_name>`
- Never commit raw CSV or `.out` result files — only MLflow-logged metrics
- The MLflow tracking URI on Narval is `file://$SCRATCH/mlruns` — set `MLFLOW_TRACKING_URI` in your env

## MLflow experiment naming convention

```
kernelserve/<cluster>/<kernel_name>
  e.g. kernelserve/narval/rms_norm
       kernelserve/narval/fused_attn
       kernelserve/nibi/rms_norm   (Phase 2 only)
```

## Viewing results

```bash
# On Narval (after salloc or from a login node)
export MLFLOW_TRACKING_URI=file://$SCRATCH/mlruns
mlflow ui --port 5000 &
# then ssh-tunnel: ssh -L 5000:localhost:5000 narval.alliancecan.ca
```
