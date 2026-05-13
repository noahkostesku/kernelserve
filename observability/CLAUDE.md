# CLAUDE.md — observability/

## What lives here

- `otel/` — OpenTelemetry instrumentation and collector config
- `prometheus/` — Prometheus scrape config
- `grafana/` — Grafana dashboard definitions

## Rules

- All spans must include `kernel.name` and `sm.target` attributes
- Metric names must be prefixed `kernelserve_` (e.g. `kernelserve_kernel_latency_seconds`)
- No PII in span attributes — do not log user IDs, IP addresses, or request payloads
- The OTel collector config must export to both Prometheus and the logging exporter in dev
- Grafana dashboard JSON must be committed so dashboards are reproducible (no manual click-ops)

## Key metrics

| Metric | Type | Description |
|---|---|---|
| `kernelserve_kernel_latency_seconds` | Histogram | Per-kernel GPU execution time |
| `kernelserve_request_duration_seconds` | Histogram | End-to-end Ray Serve request latency |
| `kernelserve_batch_size` | Histogram | Batch size distribution at Triton |
| `kernelserve_gpu_utilization_percent` | Gauge | GPU utilization from nvidia-smi |

## Starting the collector locally

```bash
# Install the OTel collector contrib distro
# https://opentelemetry.io/docs/collector/installation/
otelcol --config observability/otel/collector.yaml
```
