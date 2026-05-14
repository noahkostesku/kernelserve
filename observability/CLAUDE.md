# observability/CLAUDE.md

## Rules

- Every inference span must include: `backend_name`, `batch_size`, `input_seq_len`, `kernel_variant`, `kernel.name`, `sm.target`
- Sample GPU metrics (`util%`, `mem_used_gb`) at span start and end
- OTel collector: always export to Jaeger (local); also export to OTLP endpoint if `OTEL_EXPORTER_OTLP_ENDPOINT` is set
- Prometheus scrape intervals: 15 s for GPU metrics, 5 s for request metrics
- Grafana dashboard JSON is the source of truth — never configure dashboards via the UI
- Metric names must be prefixed `kernelserve_`
- No PII in span attributes — no user IDs, IPs, or request payloads

## Key metrics

| Metric | Type |
|---|---|
| `kernelserve_kernel_latency_seconds` | Histogram |
| `kernelserve_request_duration_seconds` | Histogram |
| `kernelserve_batch_size` | Histogram |
| `kernelserve_gpu_utilization_percent` | Gauge |
