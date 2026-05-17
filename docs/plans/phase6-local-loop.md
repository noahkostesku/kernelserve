# Phase 6 — Local Loop

## Goal

Make `ks compare` the complete one-command experience. A single invocation produces:
- Terminal side-by-side comparison table (existing)
- MLflow run logged to a local SQLite database
- OTel spans written to a local JSONL file
- The exact `mlflow ui` command to paste into a shell

No Docker, no flags, no manual setup.

## Design decisions

| Question | Decision |
|---|---|
| MLflow UI | Print command only — no auto-start |
| SQLite DB | `~/.kernelserve/mlflow.db`; override via `KERNELSERVE_MLFLOW_DB` |
| Process lifecycle | `ks compare` exits immediately after printing |
| OTel output | On by default; `KERNELSERVE_OTEL=0` disables |

## New / changed files

| File | Change |
|---|---|
| `kernelserve/otel.py` | `FileSpanExporter` + `get_tracer()` |
| `kernelserve/cli/_mlflow_uri.py` | Shared SQLite URI helper |
| `kernelserve/cli/compare.py` | MLflow logging, OTel spans, mlflow ui hint |
| `kernelserve/cli/bench.py` | Switch `--log-mlflow` to SQLite URI |

## Expected terminal output

```
WARNING: Running in CPU mock mode ...

kernel=rms_norm  batch=128  hidden_dim=512
---------------------------------------------------
backend          p50_us     p99_us       GB/s  max_abs_err
---------------------------------------------------
cpu_ref            42.10      55.30       0.25     0.00e+00
pytorch             8.91      11.02       1.21     0.00e+00
triton               nan        nan        nan          nan

Logged to MLflow (kernelserve/rms_norm/cpu_ref/local/2026-05).
OTel spans → /Users/<you>/.kernelserve/traces.jsonl

To explore results:
  mlflow ui --backend-store-uri sqlite:///Users/<you>/.kernelserve/mlflow.db
```

## Verification

```bash
# Smoke test
KERNELSERVE_DEVICE=cpu uv run ks compare --kernel rms_norm

# Artifacts exist
ls -lh ~/.kernelserve/
tail -1 ~/.kernelserve/traces.jsonl | python3 -m json.tool

# OTel opt-out
KERNELSERVE_DEVICE=cpu KERNELSERVE_OTEL=0 uv run ks compare --kernel rms_norm

# Custom DB path
KERNELSERVE_DEVICE=cpu KERNELSERVE_MLFLOW_DB=/tmp/test.db uv run ks compare --kernel rms_norm

# Full test suite
make test
```
