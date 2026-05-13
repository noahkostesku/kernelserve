---
description: Run the full benchmark suite for a named kernel against all three implementations (cuda-oxide, Triton, PyTorch) and log results to MLflow
argument-hint: <kernel-name> [--batch-size N] [--seq-len N]
allowed-tools: [Bash, Read, Grep]
---

# /bench — Kernel Benchmark Runner

Kernel: $ARGUMENTS

## Steps

1. Parse kernel name and optional `--batch-size` / `--seq-len` from $ARGUMENTS (defaults: batch=8, seq_len=512)
2. Verify the kernel exists:
   - Rust: `grep -r "<kernel_name>" kernels/cuda_oxide/src/`
   - Triton: `ls kernels/triton/<kernel_name>.py`
3. Build the Rust crate: `cd kernels/cuda_oxide && cargo build --release`
4. Run pytest benchmark suite:
   ```bash
   pytest tests/benchmark/ -m gpu -k "<kernel_name>" --benchmark-json=/tmp/bench_<kernel_name>.json
   ```
5. Log results to MLflow:
   ```bash
   python experiments/mlflow_setup.py log \
     --kernel <kernel_name> \
     --results /tmp/bench_<kernel_name>.json \
     --commit $(git rev-parse --short HEAD)
   ```
6. Print a summary table of p50/p99 latency and throughput for each implementation
7. If any implementation is >2× slower than the fastest, flag it with a warning

## Output format

```
Kernel: <name>  |  Commit: <sha>  |  GPU: A100 40GB (sm_80)
─────────────────────────────────────────────────────────────
impl          │ p50 (μs) │ p99 (μs) │ throughput (GFLOPS)
──────────────┼──────────┼──────────┼────────────────────
cuda-oxide    │   <val>  │   <val>  │       <val>
triton        │   <val>  │   <val>  │       <val>
pytorch       │   <val>  │   <val>  │       <val>
```

MLflow run logged to experiment: kernelserve/<kernel_name>
