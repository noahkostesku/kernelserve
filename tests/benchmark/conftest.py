# pytest fixtures for GPU benchmark tests.
#
# All GPU fixtures skip automatically on hosts without a CUDA device.

from __future__ import annotations

import os
import subprocess

import mlflow
import pytest


@pytest.fixture(scope="session")
def gpu_device() -> str:
    """Return the CUDA device string, or skip if no GPU is available."""
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available")
        return f"cuda:{torch.cuda.current_device()}"
    except ImportError:
        pytest.skip("torch not installed")


@pytest.fixture(scope="function")
def mlflow_run(request: pytest.FixtureRequest):
    """Start an MLflow run scoped to the test, end it when the test finishes.

    The run is tagged with the test name and git SHA for traceability.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file://./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    git_sha = _get_git_sha()
    with mlflow.start_run(run_name=request.node.name) as run:
        mlflow.set_tags({"test_name": request.node.name, "git_sha": git_sha})
        yield run


@pytest.fixture(scope="session")
def triton_client():
    """Connect to a local Triton Inference Server, or skip if not running."""
    try:
        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient("localhost:8001")
        if not client.is_server_live():
            pytest.skip("Triton Inference Server is not running at localhost:8001")
        return client
    except Exception:
        pytest.skip("Could not connect to Triton Inference Server")


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
