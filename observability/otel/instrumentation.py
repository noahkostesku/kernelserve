# OpenTelemetry instrumentation for KernelServe.
#
# Usage:
#   from observability.otel.instrumentation import create_tracer, trace_kernel
#
#   tracer = create_tracer("serving")
#
#   @trace_kernel("rms_norm")
#   def run_rms_norm(input_tensor):
#       ...

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "kernelserve")


def _build_provider(service_name: str) -> TracerProvider:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def create_tracer(service_name: str = _SERVICE_NAME) -> trace.Tracer:
    """Create and register a TracerProvider, returning a Tracer."""
    provider = _build_provider(service_name)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def trace_kernel(kernel_name: str, sm_target: str = "sm_80") -> Callable:
    """Decorator that wraps a function in an OTel span with kernel metadata."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"kernel.{kernel_name}") as span:
                span.set_attribute("kernel.name", kernel_name)
                span.set_attribute("sm.target", sm_target)
                # TODO: add batch_size and seq_len attributes from args/kwargs
                return fn(*args, **kwargs)

        return wrapper

    return decorator
