# MLflow tracking setup and run logging helpers for KernelServe.
#
# Used by:
#   - slurm/bench_job.sh  (via CLI: python experiments/mlflow_setup.py log ...)
#   - tests/benchmark/conftest.py (via mlflow_run fixture)

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow


def setup_tracking(cluster: str = "narval") -> None:
    """Configure the MLflow tracking URI.

    On Narval, results go to $SCRATCH/mlruns to avoid filling $HOME quota.
    Locally, defaults to ./mlruns.
    """
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"file://{os.environ.get('SCRATCH', '.')}/mlruns",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"kernelserve/{cluster}/default")


def log_kernel_run(
    kernel_name: str,
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    cluster: str = "narval",
) -> str:
    """Log a single benchmark run to MLflow and return the run ID.

    Args:
        kernel_name: Name of the kernel (e.g. "rms_norm")
        metrics:     Dict of metric name → float value
        tags:        Optional extra tags (merged with standard tags)
        cluster:     HPC cluster identifier ("narval" or "nibi")

    Returns:
        MLflow run ID string
    """
    setup_tracking(cluster)
    mlflow.set_experiment(f"kernelserve/{cluster}/{kernel_name}")

    git_sha = _get_git_sha()
    base_tags = {"kernel_name": kernel_name, "cluster": cluster, "git_sha": git_sha}
    if tags:
        base_tags.update(tags)

    with mlflow.start_run() as run:
        mlflow.set_tags(base_tags)
        mlflow.log_metrics(metrics)
        return run.info.run_id


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KernelServe MLflow logging CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    log_cmd = sub.add_parser("log", help="Log benchmark results from a JSON file")
    log_cmd.add_argument("--kernel", required=True)
    log_cmd.add_argument("--results", required=True, type=Path)
    log_cmd.add_argument("--commit", default=None)
    log_cmd.add_argument("--cluster", default="narval")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.command == "log":
        results_data: dict[str, Any] = json.loads(args.results.read_text())
        # TODO: parse pytest-benchmark JSON format into flat metrics dict
        # Expected keys: p50_us, p99_us, throughput_gflops, gpu_util_pct, impl
        metrics: dict[str, float] = {}
        tags: dict[str, str] = {}
        if args.commit:
            tags["git_sha"] = args.commit

        run_id = log_kernel_run(args.kernel, metrics, tags=tags, cluster=args.cluster)
        print(f"Logged run {run_id} for kernel '{args.kernel}' on {args.cluster}")
