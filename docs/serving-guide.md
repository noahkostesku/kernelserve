# Serving Guide

## Architecture recap

```
Locust → Ray Serve (HTTP :8080) → Triton gRPC (:8001)
                                     ├── cuda_oxide_backend
                                     └── triton_native_backend
```

## NVIDIA Triton Inference Server — local Docker quickstart

```bash
# TODO: pin the exact tag after team review
# Check available tags at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
TRITON_TAG="24.04-py3"

docker run --gpus all --rm \
  -p 8000:8000 \   # HTTP
  -p 8001:8001 \   # gRPC
  -p 8002:8002 \   # metrics
  -v $(pwd)/serving/triton_backends:/models \
  nvcr.io/nvidia/tritonserver:${TRITON_TAG} \
  tritonserver --model-repository=/models --log-verbose=1
```

Verify Triton is live:
```bash
curl http://localhost:8000/v2/health/ready
```

## config.pbtxt reference

Key fields in each backend's `config.pbtxt`:

| Field | Meaning |
|---|---|
| `name` | Must match the directory name under `triton_backends/` |
| `backend` | `"python"` for Python backends |
| `max_batch_size` | Max requests in one batch (must match `sample_batch.json`) |
| `dims` | Tensor shape **excluding** the batch dimension |
| `instance_group[].kind` | `KIND_GPU` to pin to GPU, `KIND_CPU` for CPU-only |

## Adding a new backend

1. Create `serving/triton_backends/<backend_name>/config.pbtxt`
2. Create `serving/triton_backends/<backend_name>/1/model.py`
3. Implement `initialize()`, `execute()`, `finalize()` in `model.py`
4. Update `serving/load_gen/payloads/sample_batch.json` if tensor shapes changed
5. Restart Triton — it hot-reloads models when `--model-control-mode=poll` is set

## Ray Serve — local launch

```bash
python serving/ray_serve/deployment.py
# Serves at http://localhost:8080
# Health check: GET http://localhost:8080/healthz
```

To change which Triton backend is used:
```bash
TRITON_MODEL_NAME=triton_native_backend python serving/ray_serve/deployment.py
```

## Running the load test

```bash
cd serving/load_gen
locust -f locustfile.py \
  --host http://localhost:8080 \
  --headless -u 20 -r 4 --run-time 120s
```

Results are printed to stdout and can be exported to CSV with `--csv=results`.

## On Narval (SLURM)

Start the serving job:
```bash
sbatch slurm/serving_job.sh
```

The job starts Triton + Ray Serve and keeps them alive for 8 hours. Monitor with:
```bash
tail -f $SCRATCH/kernelserve-logs/<job_id>-triton.log
tail -f $SCRATCH/kernelserve-logs/<job_id>-ray.log
```
