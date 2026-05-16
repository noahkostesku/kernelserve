# OpenTelemetry instrumentation for KernelServe.
#
# Usage:
#   from observability.otel.instrumentation import create_tracer, create_meter, trace_kernel
#
#   tracer = create_tracer("serving")
#   meter  = create_meter("serving")
#
#   @trace_kernel("rms_norm")
#   def run_rms_norm(input_tensor):
#       ...

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable
from typing import IO, Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

_OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "kernelserve")


def _open_span_file() -> IO[str]:
    """Open the span output file, falling back to stdout."""
    span_file = os.environ.get("OTEL_SPAN_FILE")
    if span_file:
        return open(span_file, "w")  # noqa: SIM115
    return sys.stdout


def _build_provider(service_name: str) -> TracerProvider:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        exporter: Any = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        file_exporter = ConsoleSpanExporter(out=_open_span_file())
        provider.add_span_processor(SimpleSpanProcessor(file_exporter))
    return provider


def create_tracer(service_name: str = _SERVICE_NAME) -> trace.Tracer:
    """Create and register a TracerProvider, returning a Tracer."""
    provider = _build_provider(service_name)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def create_meter(service_name: str = _SERVICE_NAME) -> metrics.Meter:
    """Create and register a MeterProvider, returning a Meter."""
    resource = Resource.create({"service.name": service_name})
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        exporter = OTLPMetricExporter(endpoint=_OTLP_ENDPOINT, insecure=True)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
    else:
        provider = MeterProvider(resource=resource)  # no-op exporter
    metrics.set_meter_provider(provider)
    return metrics.get_meter(service_name)


def trace_kernel(kernel_name: str, sm_target: str = "sm_80") -> Callable:
    """Decorator that wraps a function in an OTel span with kernel metadata.

    Optional keyword arguments ``batch_size`` and ``seq_len`` may be passed at
    call time; if present they are recorded as span attributes and then removed
    from kwargs before forwarding to the wrapped function.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            batch_size: int | None = kwargs.pop("batch_size", None)
            seq_len: int | None = kwargs.pop("seq_len", None)
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"kernel.{kernel_name}") as span:
                span.set_attribute("kernel.name", kernel_name)
                span.set_attribute("sm.target", sm_target)
                if batch_size is not None:
                    span.set_attribute("kernel.batch_size", batch_size)
                if seq_len is not None:
                    span.set_attribute("kernel.seq_len", seq_len)
                return fn(*args, **kwargs)

        return wrapper

    return decorator
