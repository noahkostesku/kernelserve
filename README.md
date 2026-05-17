# KernelServe

Custom GPU kernel benchmarking platform. cuda-oxide Rust kernels vs Triton JIT vs PyTorch baselines, served via NVIDIA Triton Inference Server and Ray Serve, tracked with MLflow and OpenTelemetry.

---

## Benchmark Results

### Narval A100 40 GB (sm\_80)

| Shape | PyTorch (µs) | Triton (µs) | cuda-oxide (µs) | Throughput (GB/s) |
|---|---|---|---|---|
| 256×512 | 65.2 | 65.7 | 19.5 | 54.0 |
| 2048×4096 | 215.8 | 203.6 | 111.6 | 601.4 |
| 4096×8192 | 723.7 | 724.2 | 388.6 | 690.9 |

### Nibi H100 80 GB (sm\_90)

| Shape | PyTorch (µs) | Triton (µs) | cuda-oxide (µs) | Throughput (GB/s) |
|---|---|---|---|---|
| 256×512 | 34.4 | 34.7 | 20.4 | 51.4 |
| 2048×4096 | 113.5 | 113.7 | 65.6 | 1022.9 |
| 4096×8192 | 375.2 | 375.0 | 190.5 | 1409.2 |

> **cuda-oxide peaks at 1409 GB/s on H100 at 4096×8192 — approaching H100 HBM3 theoretical bandwidth of ~3.35 TB/s for a memory-bound kernel.**

Kernel: scalar-naive RMS-Norm with warp-shuffle reduction. MLflow experiment format: `kernelserve/rms_norm/<backend>/<cluster>/2026-05`.

---

## What This Is

KernelServe is a GPU kernel benchmarking platform that measures custom cuda-oxide Rust kernels (scalar-naive RMS-Norm with warp-shuffle reduction) against Triton JIT and PyTorch baselines across multiple tensor shapes and GPU architectures. Benchmark runs are tracked end-to-end with MLflow experiment logging, OpenTelemetry distributed tracing, pynvml GPU utilization sampling, and a Grafana dashboard for cross-backend comparison. The platform runs on Alliance Canada HPC (Narval A100 40 GB, Nibi H100 80 GB) via SLURM job submission, with a local docker-compose stack for development and dashboard validation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Kernels                             │
│   cuda-oxide (Rust)  │  Triton (Python)  │  PyTorch (ref)  │
└───────────────────────────────┬─────────────────────────────┘
                                │ bench harness
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                          Serving                            │
│   NVIDIA Triton Inference Server  │  Ray Serve (local)      │
└───────────────────────────────┬─────────────────────────────┘
                                │ metrics / traces
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Observability                          │
│  OpenTelemetry → otelcol → Prometheus → Grafana             │
│  pynvml GPU sampling          │  MLflow experiment tracking │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **Kernels:** cuda-oxide (Rust + PTX), Triton (Python JIT), PyTorch
- **Serving:** NVIDIA Triton Inference Server, Ray Serve (local mode)
- **Experiment tracking:** MLflow (`file://$SCRATCH/mlruns` on HPC)
- **Observability:** OpenTelemetry SDK, otelcol, Prometheus, Grafana, pynvml
- **HPC:** Alliance Canada SLURM — Narval (A100 sm\_80), Nibi (H100 sm\_90)
- **Build:** `cargo oxide build` (not `cargo build`), `uv` for Python packages

---

## Project Structure

```
kernelserve/
├── kernels/
│   ├── cuda_oxide/          # Rust cuda-oxide kernels + correctness tests
│   └── triton/              # Triton JIT baselines + fixture generation
├── serving/                 # Triton Inference Server backends, Ray Serve
├── experiments/             # Python benchmark harness (bench_rms_norm.py)
├── observability/
│   ├── otel/                # OTel instrumentation + collector config
│   ├── metrics/             # pynvml GPU sampler
│   ├── prometheus/          # scrape config
│   └── grafana/             # dashboard JSON + provisioning
├── slurm/                   # All SLURM job scripts
├── tests/                   # Integration tests (CPU-only, no GPU marker)
├── docs/
│   └── plans/               # Phase implementation plans
├── docker-compose.yml       # Local observability stack (dev only)
└── pyproject.toml
```

---

## Getting Started

### Local (dev)

```bash
# Start observability stack
docker compose up -d

# Run benchmark harness (CPU mock mode, no GPU required)
MLFLOW_TRACKING_URI=file:///tmp/mlruns \
  uv run python experiments/bench_rms_norm.py

# Grafana: http://localhost:3000
# Jaeger:  http://localhost:16686
# MLflow:  mlflow ui --backend-store-uri file:///tmp/mlruns
```

### HPC (Alliance Canada — Narval / Nibi)

```bash
# Load environment
module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0 python/3.11 llvm/18.1.8

# Create venv once
python -m venv .venv && source .venv/bin/activate && uv sync

# Submit benchmark job
sbatch slurm/bench_rms_norm_phase4_nibi.sh   # Nibi H100
sbatch slurm/bench_rms_norm_phase3.sh        # Narval A100

# Check results
mlflow ui --backend-store-uri file://$SCRATCH/mlruns
```

---

## Phases

| Phase | Deliverable |
|---|---|
| 1 | cuda-oxide RMS-Norm kernel (scalar-naive, warp-shuffle reduction) on Narval A100; correctness verified against PyTorch (max abs error < 1e-4) |
| 2 | Three-backend benchmark (cuda-oxide / Triton / PyTorch) across three shapes on Narval A100; 9 MLflow runs logged |
| 3 | OpenTelemetry tracing + pynvml GPU sampling in bench harness; Grafana dashboard with 5 panels; Jaeger trace export on SLURM |
| 4 | Phase 3 benchmark re-run on Nibi H100 (sm\_90); A100 vs H100 comparison; peak 1409 GB/s cuda-oxide throughput |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Compatible with [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (NVlabs Apache-2.0).
