# CLAUDE.md — serving/

## What lives here

- `triton_backends/` — NVIDIA Triton Inference Server backend definitions
- `ray_serve/` — Ray Serve deployment wrapping Triton
- `load_gen/` — Locust load generator

## Rules

- `config.pbtxt` max_batch_size must match the tensor shapes in `load_gen/payloads/sample_batch.json`
- Every backend directory must have a `config.pbtxt` at the root and a `1/model.py` for the Python backend
- Ray Serve deployments must expose a `/healthz` route
- Never put secrets (API keys, credentials) in `config.yaml` — use `os.environ.get("VAR_NAME")` with documented names
- When changing tensor shapes in a backend, update `sample_batch.json` in the same commit

## Workflow

1. Read this file
2. Backend changes: update `config.pbtxt` first → then `model.py` → verify shapes match `sample_batch.json`
3. Ray Serve changes: update `deployment.py` → then `config.yaml`
4. Load test changes: update `locustfile.py` → verify payload shape matches backend input spec

## Local testing

```bash
# Start Triton server (requires Docker + NVIDIA Container Toolkit)
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/serving/triton_backends:/models \
  nvcr.io/nvidia/tritonserver:<tag> \
  tritonserver --model-repository=/models

# Start Ray Serve
python serving/ray_serve/deployment.py

# Run load test
cd serving/load_gen && locust -f locustfile.py --headless -u 10 -r 2 --run-time 60s
```
