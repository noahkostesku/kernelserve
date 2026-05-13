# Architecture

## Overview

KernelServe is a research benchmarking platform that:

1. **Implements** GPU kernels in Rust (cuda-oxide) targeting NVIDIA A100 (sm_80, Phase 1)
2. **Compares** them against Python/Triton baselines and vanilla PyTorch
3. **Serves** kernels through NVIDIA Triton Inference Server, fronted by Ray Serve
4. **Tracks** all experiments with MLflow on Alliance Canada HPC (Narval)
5. **Observes** serving latency via OpenTelemetry → Prometheus → Grafana

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Load Generator (Locust)                                     │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────────────┐
│  Ray Serve (serving/ray_serve/)                             │
│  - autoscaling frontend                                     │
│  - /healthz endpoint                                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ gRPC
┌─────────────────────────▼───────────────────────────────────┐
│  NVIDIA Triton Inference Server                             │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │ cuda_oxide_backend   │  │ triton_native_backend        │ │
│  │ (PTX / Rust kernels) │  │ (Python/Triton kernels)      │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Kernel Layer                                               │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │ kernels/cuda_oxide/  │  │ kernels/triton/              │ │
│  │ Rust + cuda-oxide    │  │ Python + Triton              │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Observability                                              │
│  OTel Collector → Prometheus → Grafana                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Experiment Tracking (MLflow)                               │
│  Stored at $SCRATCH/mlruns on Narval                        │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. A SLURM benchmark job (`slurm/bench_job.sh`) builds the Rust kernel and runs pytest-benchmark
2. pytest-benchmark records latency distributions for each impl (cuda-oxide, Triton, PyTorch)
3. Results are logged to MLflow via `experiments/mlflow_setup.py`
4. For serving benchmarks, Locust sends requests through Ray Serve → Triton → kernel
5. OTel spans flow from Ray Serve and Triton model.py → collector → Prometheus

## Technology Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Kernel language | Rust (cuda-oxide) | Type safety, memory safety, no undefined behavior |
| Baseline | Python/Triton | Industry standard, fair GPU-language comparison |
| Serving | NVIDIA Triton + Ray Serve | Triton for GPU batching, Ray for scaling/routing |
| Experiment tracking | MLflow | Self-hostable, works offline on HPC clusters |
| Observability | OTel + Prometheus + Grafana | Open standards, works in air-gapped environments |
| HPC | Alliance Canada (SLURM) | Available to the team; A100 access on Narval |

## Phase 1 vs Phase 2 Scope

| | Phase 1 (current) | Phase 2 |
|---|---|---|
| Cluster | Narval | Nibi |
| GPU | A100 40 GB | H100 80 GB |
| sm target | sm_80 | sm_90 |
| LLVM requirement | ≥ 18 | ≥ 21 |
| Kernels | rms_norm, fused_attn (stubs) | TBD |
| Hopper features | none | TMA, wgmma |
