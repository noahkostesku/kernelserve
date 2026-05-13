---
description: Verify the OpenTelemetry collector is running and traces are flowing from the serving layer
argument-hint: [--endpoint http://localhost:4317]
allowed-tools: [Bash, Read]
---

# /otel-check — OpenTelemetry Health Check

## Steps

1. Parse optional `--endpoint` from $ARGUMENTS (default: `http://localhost:4317`)
2. Check if the OTel collector process is running:
   ```bash
   pgrep -f otelcol || echo "Collector not running"
   ```
3. Validate the collector config:
   ```bash
   cat observability/otel/collector.yaml
   ```
4. Send a test span:
   ```bash
   python3 -c "
   from observability.otel.instrumentation import create_tracer
   tracer = create_tracer('otel-check')
   with tracer.start_as_current_span('health-check') as span:
       span.set_attribute('check.status', 'ok')
   print('Test span sent')
   "
   ```
5. Query Prometheus for the `kernelserve_kernel_latency_seconds` metric:
   ```bash
   curl -s 'http://localhost:9090/api/v1/query?query=kernelserve_kernel_latency_seconds' | python3 -m json.tool | head -20
   ```
6. Report: collector running (yes/no) | test span sent (yes/no) | Prometheus metric present (yes/no)
