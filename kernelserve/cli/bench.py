from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import argparse

_GPU_DEFAULTS = dict(warmup=100, iters=1000, batch=2048, hidden_dim=4096)
_CPU_DEFAULTS = dict(warmup=3,   iters=10,   batch=128,  hidden_dim=512)


def _is_cpu_mode(args: argparse.Namespace) -> bool:
    if getattr(args, "fast", False):
        return True
    if os.environ.get("KERNELSERVE_DEVICE", "").lower() == "cpu":
        return True
    return not torch.cuda.is_available()


def _detect_backend() -> str:
    if os.environ.get("KERNELSERVE_DEVICE", "").lower() == "cpu":
        return "cpu_ref"
    return "cuda_oxide" if torch.cuda.is_available() else "cpu_ref"


def _detect_cluster() -> str:
    if os.environ.get("SLURM_JOB_ID"):
        return os.environ.get("CLUSTER", "narval")
    return os.environ.get("CLUSTER", "local")


def run_bench(args: argparse.Namespace) -> None:
    import kernelserve

    cpu_mode = _is_cpu_mode(args)
    defaults = _CPU_DEFAULTS if cpu_mode else _GPU_DEFAULTS

    warmup    = args.warmup     if args.warmup     is not None else defaults["warmup"]
    iters     = args.iters      if args.iters      is not None else defaults["iters"]
    batch     = args.batch      if args.batch      is not None else defaults["batch"]
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else defaults["hidden_dim"]

    if cpu_mode:
        print(
            "WARNING: Running in CPU mock mode — results are not representative "
            "of GPU performance.",
            file=sys.stderr,
        )

    backend = _detect_backend()
    cluster = _detect_cluster()

    x = torch.randn(batch, hidden_dim, dtype=torch.float32)
    w = torch.ones(hidden_dim, dtype=torch.float32)

    for _ in range(warmup):
        kernelserve.rms_norm(x, w)

    times_us: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        kernelserve.rms_norm(x, w)
        times_us.append((time.perf_counter() - t0) * 1e6)

    times_us.sort()
    p50 = times_us[iters // 2]
    p99 = times_us[int(iters * 0.99)]
    bytes_accessed = (2 * batch * hidden_dim + hidden_dim) * 4
    throughput_gbs = bytes_accessed / (p50 / 1e6) / 1e9

    result = {
        "backend": backend,
        "batch": batch,
        "hidden_dim": hidden_dim,
        "p50_us": round(p50, 3),
        "p99_us": round(p99, 3),
        "throughput_gbs": round(throughput_gbs, 2),
    }
    print(json.dumps(result))

    if args.log_mlflow:
        import mlflow

        from kernelserve.cli._mlflow_uri import mlflow_sqlite_uri

        month = datetime.now().strftime("%Y-%m")
        uri, is_sqlite = mlflow_sqlite_uri()
        if not is_sqlite:
            print(
                f"WARNING: sqlite3 unavailable — MLflow falling back to {uri}",
                file=sys.stderr,
            )
        mlflow.set_tracking_uri(uri)
        experiment_name = f"kernelserve/{args.kernel}/{backend}/{cluster}/{month}"
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
                "backend": backend,
                "cluster": cluster,
            })
            mlflow.log_metrics({
                "p50_us": result["p50_us"],
                "p99_us": result["p99_us"],
                "throughput_gbs": result["throughput_gbs"],
            })
        print(f"Logged to MLflow experiment: {experiment_name}")
