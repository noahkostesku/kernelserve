"""Benchmark RMS-Norm across cuda-oxide, Triton, and PyTorch on Narval A100.

Logs to MLflow: kernelserve/rms_norm/<backend>/narval/2026-05
Run via: srun python experiments/bench_rms_norm.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import mlflow
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kernels.triton.rms_norm import rms_norm_torch, rms_norm_triton  # type: ignore[import]

SHAPES = [(256, 512), (2048, 4096), (4096, 8192)]
WARMUP = 100
ITERS = 1000
CLUSTER = "narval"
MONTH = "2026-05"


class BenchResult(NamedTuple):
    backend: str
    batch: int
    hidden_dim: int
    p50_us: float
    p99_us: float
    throughput_gbs: float


def _throughput_gbs(batch: int, hidden_dim: int, p50_us: float) -> float:
    bytes_total = (2 * batch * hidden_dim + hidden_dim) * 4
    return bytes_total / (p50_us / 1e6) / 1e9


def _bench_torch_fn(
    fn: object, x: torch.Tensor, weight: torch.Tensor
) -> tuple[float, float]:
    """Time a GPU callable with CUDA events. Returns (p50_us, p99_us)."""
    import typing

    callable_fn = typing.cast(typing.Callable[..., object], fn)
    for _ in range(WARMUP):
        callable_fn(x, weight)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()  # type: ignore[no-untyped-call]
        callable_fn(x, weight)
        end.record()  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))  # type: ignore[no-untyped-call]

    times_us = np.array(times_ms, dtype=np.float64) * 1000.0
    return float(np.percentile(times_us, 50)), float(np.percentile(times_us, 99))


def bench_pytorch(batch: int, hidden_dim: int) -> BenchResult:
    x = torch.randn(batch, hidden_dim, device="cuda", dtype=torch.float32)
    w = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    p50, p99 = _bench_torch_fn(rms_norm_torch, x, w)
    gbs = _throughput_gbs(batch, hidden_dim, p50)
    return BenchResult("pytorch", batch, hidden_dim, p50, p99, gbs)


def bench_triton(batch: int, hidden_dim: int) -> BenchResult:
    x = torch.randn(batch, hidden_dim, device="cuda", dtype=torch.float32)
    w = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    p50, p99 = _bench_torch_fn(rms_norm_triton, x, w)
    gbs = _throughput_gbs(batch, hidden_dim, p50)
    return BenchResult("triton", batch, hidden_dim, p50, p99, gbs)


def bench_cuda_oxide(batch: int, hidden_dim: int) -> BenchResult:
    bin_path = os.environ.get(
        "CUDA_OXIDE_BIN",
        str(
            Path(__file__).resolve().parent.parent
            / "kernels"
            / "rms_norm_standalone"
            / "target"
            / "release"
            / "rms_norm_standalone"
        ),
    )
    result = subprocess.run(
        [bin_path, "--batch", str(batch), "--hidden-dim", str(hidden_dim), "--bench"],
        capture_output=True,
        text=True,
        check=True,
    )
    json_line = next(
        line for line in result.stdout.splitlines()
        if line.strip().startswith("{")
    )
    data: dict[str, float] = json.loads(json_line)
    p50 = float(data["p50_us"])
    p99 = float(data["p99_us"])
    gbs = float(data["throughput_gbs"])
    return BenchResult("cuda_oxide", batch, hidden_dim, p50, p99, gbs)


def log_result(result: BenchResult) -> None:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    exp_name = f"kernelserve/rms_norm/{result.backend}/{CLUSTER}/{MONTH}"
    experiment = mlflow.set_experiment(exp_name)
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.set_tag("git_sha", git_sha)
        mlflow.log_params({"batch": result.batch, "hidden_dim": result.hidden_dim})
        mlflow.log_metrics(
            {
                "latency_p50_us": result.p50_us,
                "latency_p99_us": result.p99_us,
                "throughput_gbs": result.throughput_gbs,
            }
        )


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    from observability.metrics.gpu import sample_gpu_metrics
    from observability.otel.instrumentation import create_meter, create_tracer

    tracer = create_tracer("kernelserve-bench")
    meter = create_meter("kernelserve-bench")
    p50_gauge = meter.create_gauge("kernelserve_bench_latency_p50_us", unit="us")
    p99_gauge = meter.create_gauge("kernelserve_bench_latency_p99_us", unit="us")
    tput_gauge = meter.create_gauge("kernelserve_bench_throughput_gbs", unit="GB/s")
    gpu_util_gauge = meter.create_gauge("kernelserve_gpu_utilization_percent", unit="%")
    gpu_mem_gauge = meter.create_gauge("kernelserve_gpu_memory_used_gb", unit="GB")

    benchers = [
        ("pytorch", bench_pytorch),
        ("triton", bench_triton),
        ("cuda_oxide", bench_cuda_oxide),
    ]

    for batch, hidden_dim in SHAPES:
        print(f"\n=== shape {batch}x{hidden_dim} ===")
        for name, bencher in benchers:
            print(f"  [{name}] benchmarking...", end=" ", flush=True)
            attrs = {"backend_name": name, "shape": f"{batch}x{hidden_dim}"}
            with tracer.start_as_current_span(f"bench.rms_norm.{name}") as span:
                span.set_attribute("backend_name", name)
                span.set_attribute("batch_size", batch)
                span.set_attribute("input_seq_len", hidden_dim)
                span.set_attribute("kernel_variant", "rms_norm")
                span.set_attribute("kernel.name", "rms_norm")
                span.set_attribute("sm.target", os.environ.get("CUDA_ARCH", "sm_80"))
                span.set_attribute("shape", f"{batch}x{hidden_dim}")
                span.add_event("gpu.sample.start", attributes=sample_gpu_metrics())
                result = bencher(batch, hidden_dim)
                span.add_event("gpu.sample.end", attributes=sample_gpu_metrics())
                span.set_attribute("latency_p50_us", result.p50_us)
                span.set_attribute("latency_p99_us", result.p99_us)
                span.set_attribute("throughput_gbs", result.throughput_gbs)
            log_result(result)
            print(
                f"p50={result.p50_us:.1f}µs  p99={result.p99_us:.1f}µs  "
                f"tput={result.throughput_gbs:.1f} GB/s"
            )
            gauge_attrs = attrs | {"kernel": "rms_norm"}
            p50_gauge.set(result.p50_us, gauge_attrs)
            p99_gauge.set(result.p99_us, gauge_attrs)
            tput_gauge.set(result.throughput_gbs, gauge_attrs)
            gpu_snap = sample_gpu_metrics()
            if gpu_snap:
                gpu_util_gauge.set(gpu_snap["gpu.util_pct"], attrs)
                gpu_mem_gauge.set(gpu_snap["gpu.mem_used_gb"], attrs)

    print("\nDone. Results logged to MLflow.")


if __name__ == "__main__":
    main()
