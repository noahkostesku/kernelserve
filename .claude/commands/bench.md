---
description: Run the full benchmark suite across all three backends (cuda-oxide, Triton, PyTorch) for a specified kernel and log to MLflow
argument-hint: [kernel-name]
allowed-tools: [Bash, Read]
---

# /bench — Kernel Benchmark Runner

## Steps

1. If $ARGUMENTS is empty, ask: "Which kernel to benchmark? (rms_norm / fused_attn / all)"
   Otherwise use $ARGUMENTS as the kernel name.

2. Check GPU availability:
   ```bash
   nvidia-smi
   ```
   If this fails, abort: "No GPU detected — benchmark requires a GPU."

3. Check if Triton Inference Server is running:
   ```bash
   curl -s --max-time 3 http://localhost:8000/v2/health/ready
   ```
   If not ready, start it:
   ```bash
   docker compose up -d triton
   ```
   Poll every 3 s up to 30 s. Abort if still not ready.

4. For each backend in `cuda_oxide triton pytorch`, run perf_analyzer:
   ```bash
   perf_analyzer -m <kernel>_<backend> -u localhost:8001 \
     --concurrency-range 1 \
     --measurement-interval 5000 \
     --warmup-request-count 3 \
     --measurement-request-count 100 \
     --percentile 99
   ```
   Parse p50, p99 latency (ms) and throughput (infer/s) from stdout.
   If `all` was selected, repeat for every kernel.

5. Log results to MLflow. Use experiment name format exactly:
   `kernelserve/<kernel>/<backend>/narval/<YYYY-MM>`
   Set URI to `file://$SCRATCH/mlruns` when `$SCRATCH` is set, else `./mlruns`.
   Tag every run:
   ```python
   mlflow.set_tag("git_sha", subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip())
   ```

6. Print a markdown comparison table:

   ```
   | Backend    | p50 (ms) | p99 (ms) | Throughput (infer/s) |
   |------------|----------|----------|----------------------|
   | cuda_oxide |          |          |                      |
   | triton     |          |          |                      |
   | pytorch    |          |          |                      |
   ```

   Flag any backend that is >2× slower than the fastest with a warning.

7. Ask: "Open Grafana dashboard at http://localhost:3000/d/kernelserve? (y/n)"
   Do not open it unless the user says yes.
