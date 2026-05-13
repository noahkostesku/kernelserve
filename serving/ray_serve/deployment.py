# Ray Serve deployment wrapping the NVIDIA Triton Inference Server.
#
# Provides an HTTP frontend that forwards requests to Triton via gRPC,
# enabling Ray's autoscaling and routing on top of Triton's GPU batching.

from __future__ import annotations

import os

import tritonclient.grpc as grpcclient
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
)
class KernelServeDeployment:
    """Ray Serve deployment that proxies requests to a Triton Inference Server."""

    def __init__(self) -> None:
        triton_url = os.environ.get("TRITON_GRPC_URL", "localhost:8001")
        # TODO: make connection retry-able with exponential backoff
        self._client = grpcclient.InferenceServerClient(url=triton_url)
        self._model_name = os.environ.get("TRITON_MODEL_NAME", "cuda_oxide_backend")

    async def __call__(self, request: Request) -> JSONResponse:
        """Forward an inference request to Triton and return the result."""
        # TODO: parse request body, build grpcclient.InferInput objects
        # TODO: call self._client.infer(self._model_name, inputs=[...])
        # TODO: extract output tensors and serialize to JSON response
        return JSONResponse({"status": "not_implemented"}, status_code=501)

    async def healthz(self) -> JSONResponse:
        """Health check endpoint required by serving/CLAUDE.md."""
        try:
            live = self._client.is_server_live()
            return JSONResponse({"status": "ok", "triton_live": live})
        except Exception as exc:
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)


app = KernelServeDeployment.bind()

if __name__ == "__main__":
    serve.run(app, host="0.0.0.0", port=8080)
