# serving/CLAUDE.md

## Rules

- `config.pbtxt` must set `backend: "python"` (not `"pytorch"`), `max_batch_size`, and a `dynamic_batching` block
- Never import `ray` at module level in `model.py` — Triton loads these files without Ray
- Ray Serve: `@serve.deployment(num_replicas=1)` default; override with `KERNELSERVE_REPLICAS` env var
- All endpoints must return `latency_ms` in response metadata
- Load test every serving change before merging: `cd load_gen && locust -f locustfile.py --headless -u 10 -r 2 --run-time 60s`
- Tensor shapes in `config.pbtxt` must match `load_gen/payloads/sample_batch.json` — update in the same commit
- Every backend dir needs `config.pbtxt` at root and `1/model.py`
- Ray Serve deployments must expose `/healthz`
- Never hardcode secrets in `config.yaml` — use `os.environ.get("VAR_NAME")`

## Workflow order

- Backend change: `config.pbtxt` → `model.py` → verify shapes match `sample_batch.json`
- Ray Serve change: `deployment.py` → `config.yaml`
