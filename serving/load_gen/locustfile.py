# Locust load generator for the KernelServe Ray Serve endpoint.
#
# Run locally:
#   locust -f serving/load_gen/locustfile.py --headless -u 10 -r 2 --run-time 60s
#
# Run on cluster (after serving_job.sh is up):
#   locust -f serving/load_gen/locustfile.py --host http://<ray-serve-ip>:8080 ...

import json
import os
from pathlib import Path

from locust import HttpUser, between, task

PAYLOAD_PATH = Path(__file__).parent / "payloads" / "sample_batch.json"


class KernelServeUser(HttpUser):
    """Simulates a client sending inference requests to KernelServe."""

    wait_time = between(0.01, 0.1)
    host = os.environ.get("KERNELSERVE_HOST", "http://localhost:8080")

    def on_start(self) -> None:
        self._payload = json.loads(PAYLOAD_PATH.read_text())

    @task(weight=10)
    def benchmark_kernel(self) -> None:
        """Send an inference request and record latency."""
        # TODO: set the correct endpoint path once Ray Serve routing is finalized
        self.client.post(
            "/infer",
            json=self._payload,
            name="POST /infer",
        )

    @task(weight=1)
    def health_check(self) -> None:
        self.client.get("/healthz", name="GET /healthz")
