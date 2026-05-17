from __future__ import annotations

import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from typing import TYPE_CHECKING

import torch

from kernelserve.cli._mlflow_uri import mlflow_sqlite_uri
from kernelserve.otel import get_tracer

if TYPE_CHECKING:
    import argparse

_COL = 16

_GPU_DEFAULTS = dict(warmup=20, iters=200, batch=2048, hidden_dim=4096)
_CPU_DEFAULTS = dict(warmup=2,  iters=5,   batch=128,  hidden_dim=512)


def _is_cpu_mode(args: argparse.Namespace) -> bool:
    if getattr(args, "fast", False):
        return True
    if os.environ.get("KERNELSERVE_DEVICE", "").lower() == "cpu":
        return True
    return not torch.cuda.is_available()


def _detect_cluster() -> str:
    if os.environ.get("SLURM_JOB_ID"):
        return os.environ.get("CLUSTER", "narval")
    return os.environ.get("CLUSTER", "local")


def _time_fn(fn: object, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()  # type: ignore[operator]
    times_us = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()  # type: ignore[operator]
        times_us.append((time.perf_counter() - t0) * 1e6)
    times_us.sort()
    return times_us[iters // 2], times_us[int(iters * 0.99)]


def run_compare(args: argparse.Namespace) -> None:
    import kernelserve

    cpu_mode = _is_cpu_mode(args)
    defaults = _CPU_DEFAULTS if cpu_mode else _GPU_DEFAULTS

    warmup     = defaults["warmup"]
    iters      = defaults["iters"]
    batch      = args.batch      if args.batch      is not None else defaults["batch"]
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else defaults["hidden_dim"]

    if cpu_mode:
        print(
            "WARNING: Running in CPU mock mode — results are not representative "
            "of GPU performance.",
            file=sys.stderr,
        )

    cluster = _detect_cluster()
    tracer = get_tracer(args.kernel)

    x = torch.randn(batch, hidden_dim, dtype=torch.float32)
    w = torch.ones(hidden_dim, dtype=torch.float32)
    bytes_accessed = (2 * batch * hidden_dim + hidden_dim) * 4

    ref = torch.nn.functional.rms_norm(x, (hidden_dim,), weight=w)
    rows: list[tuple[str, float, float, float, float]] = []

    with tracer.start_as_current_span(f"compare.{args.kernel}"):
        # cuda-oxide / cpu-ref path
        backend_name = "cuda_oxide" if (
            os.environ.get("KERNELSERVE_DEVICE", "").lower() != "cpu"
            and torch.cuda.is_available()
        ) else "cpu_ref"
        with tracer.start_as_current_span(f"backend.{backend_name}") as span:
            p50, p99 = _time_fn(lambda: kernelserve.rms_norm(x, w), warmup, iters)
            out_ks = kernelserve.rms_norm(x, w)
            max_err = float((out_ks - ref).abs().max())
            gbs = bytes_accessed / (p50 / 1e6) / 1e9
            span.set_attributes({
                "p50_us": p50, "p99_us": p99,
                "throughput_gbs": gbs, "max_abs_err": max_err,
            })
        rows.append((backend_name, p50, p99, gbs, max_err))

        # PyTorch reference
        with tracer.start_as_current_span("backend.pytorch") as span:
            p50, p99 = _time_fn(
                lambda: torch.nn.functional.rms_norm(x, (hidden_dim,), weight=w), warmup, iters
            )
            gbs = bytes_accessed / (p50 / 1e6) / 1e9
            span.set_attributes({"p50_us": p50, "p99_us": p99, "throughput_gbs": gbs, "max_abs_err": 0.0})
        rows.append(("pytorch", p50, p99, gbs, 0.0))

        # Triton (Linux + CUDA only)
        with tracer.start_as_current_span("backend.triton") as span:
            try:
                sys.path.insert(0, "kernels/triton")
                from rms_norm import rms_norm as triton_rms_norm  # type: ignore[import]
                p50, p99 = _time_fn(lambda: triton_rms_norm(x, w, eps=1e-5), warmup, iters)
                out_tri = triton_rms_norm(x, w, eps=1e-5)
                max_err_tri = float((out_tri - ref).abs().max())
                gbs_tri = bytes_accessed / (p50 / 1e6) / 1e9
                span.set_attributes({
                    "p50_us": p50, "p99_us": p99,
                    "throughput_gbs": gbs_tri, "max_abs_err": max_err_tri,
                })
                rows.append(("triton", p50, p99, gbs_tri, max_err_tri))
            except ImportError:
                rows.append(("triton", float("nan"), float("nan"), float("nan"), float("nan")))

    header = (
        f"{'backend':<{_COL}} {'p50_us':>10} {'p99_us':>10} {'GB/s':>10} {'max_abs_err':>14}"
    )
    sep = "-" * len(header)
    print(f"\nkernel={args.kernel}  batch={batch}  hidden_dim={hidden_dim}")
    print(sep)
    print(header)
    print(sep)
    for name, p50, p99, gbs, err in rows:
        print(
            f"{name:<{_COL}} {p50:>10.2f} {p99:>10.2f} {gbs:>10.2f} {err:>14.2e}"
        )
    print()

    # MLflow logging (always on for compare)
    import mlflow

    uri, is_sqlite = mlflow_sqlite_uri()
    if not is_sqlite:
        print(
            f"WARNING: sqlite3 unavailable — MLflow falling back to {uri}",
            file=sys.stderr,
        )
    month = datetime.now().strftime("%Y-%m")
    primary_backend = rows[0][0]
    experiment_name = f"kernelserve/{args.kernel}/{primary_backend}/{cluster}/{month}"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]
            ).decode().strip()
            mlflow.set_tag("git_sha", git_sha)
            mlflow.log_params({
                "batch": batch,
                "hidden_dim": hidden_dim,
                "kernel": args.kernel,
                "cluster": cluster,
            })
            for name, p50_v, p99_v, gbs_v, err_v in rows:
                mlflow.log_metrics({
                    f"p50_us_{name}": p50_v,
                    f"p99_us_{name}": p99_v,
                    f"throughput_gbs_{name}": gbs_v,
                    f"max_abs_err_{name}": err_v,
                })

    print(f"Logged to MLflow ({experiment_name}).")

    from kernelserve.otel import _otel_enabled, _traces_path
    if _otel_enabled():
        print(f"OTel spans → {_traces_path()}")

    print(f"\nTo explore results:")
    print(f"  mlflow ui --backend-store-uri {uri}")
