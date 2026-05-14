---
description: Verify the observability stack is running and receiving data
argument-hint: [--endpoint http://localhost:4317]
allowed-tools: [Bash, Read]
---

# /otel-check — Observability Stack Health Check

## Steps

1. Check Jaeger at `localhost:16686`:
   ```bash
   curl -s --max-time 3 http://localhost:16686 -o /dev/null -w "%{http_code}"
   ```
   Record ✓ if HTTP 200, ✗ otherwise.

2. Check Prometheus at `localhost:9090`:
   ```bash
   curl -s --max-time 3 http://localhost:9090/-/healthy
   ```
   Record ✓ if response is `Prometheus Server is Healthy.`, ✗ otherwise.

3. Check Grafana at `localhost:3000`:
   ```bash
   curl -s --max-time 3 http://localhost:3000/api/health | python3 -m json.tool
   ```
   Record ✓ if `"database": "ok"` is in the response, ✗ otherwise.

4. Send a test span via the OTel SDK:
   ```bash
   python3 -c "
   from opentelemetry import trace
   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import BatchSpanProcessor
   from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

   provider = TracerProvider()
   provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint='http://localhost:4317', insecure=True)))
   trace.set_tracer_provider(provider)
   tracer = trace.get_tracer('otel-check')
   with tracer.start_as_current_span('health-check') as span:
       span.set_attribute('check.status', 'ok')
   provider.force_flush()
   print('span sent')
   "
   ```
   Then confirm the span appears in Jaeger:
   ```bash
   curl -s 'http://localhost:16686/api/traces?service=otel-check&limit=1' | python3 -m json.tool | head -10
   ```
   Record ✓ if a trace is returned, ✗ if the result is empty.

5. Query Prometheus for the most recent `kernelserve_request_latency_ms` sample:
   ```bash
   curl -s 'http://localhost:9090/api/v1/query?query=kernelserve_request_latency_ms' | python3 -m json.tool
   ```
   Record ✓ if at least one result is returned, ✗ if the result list is empty.

6. Print a health table:

   ```
   Component   | Endpoint                | Status
   ------------|-------------------------|-------
   Jaeger      | localhost:16686         | ✓/✗
   Prometheus  | localhost:9090          | ✓/✗
   Grafana     | localhost:3000          | ✓/✗
   Test span   | Jaeger trace confirmed  | ✓/✗
   Metric      | request_latency_ms      | ✓/✗
   ```

   If any component shows ✗, print the raw error output below the table.
