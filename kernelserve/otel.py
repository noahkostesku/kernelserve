from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import NoOpTracer

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan


def _otel_enabled() -> bool:
    return os.environ.get("KERNELSERVE_OTEL", "1") != "0"


def _traces_path() -> Path:
    default = Path.home() / ".kernelserve" / "traces.jsonl"
    return Path(os.environ.get("KERNELSERVE_OTEL_FILE", str(default)))


class FileSpanExporter(SpanExporter):
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._path.open("a") as f:
            for span in spans:
                ctx = span.get_span_context()
                record = {
                    "name": span.name,
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "start_time_unix_nano": span.start_time,
                    "end_time_unix_nano": span.end_time,
                    "attributes": dict(span.attributes or {}),
                }
                f.write(json.dumps(record) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def get_tracer(kernel: str) -> trace.Tracer:
    if not _otel_enabled():
        trace.set_tracer_provider(trace.NoOpTracerProvider())
        return trace.get_tracer(__name__)

    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(FileSpanExporter(_traces_path())))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(f"kernelserve.{kernel}")
