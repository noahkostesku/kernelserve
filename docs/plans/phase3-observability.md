# Phase 3 Observability Plan

## Context

Phase 2 delivered 9 MLflow runs (3 backends × 3 shapes) for the RMS-Norm kernel.
The runs contain p50/p99 latency and throughput but no distributed traces, no GPU
utilization snapshots, and no live visualization. Phase 3 wires OpenTelemetry tracing
into the benchmark harness, adds pynvml GPU sampling, and builds out the Grafana
dashboard so backend/shape comparisons are visible in a single panel grid.

**Decisions from interview:**
- Tracing scope: bench harness only (`experiments/bench_rms_norm.py`) — no serving changes
- GPU metrics: pynvml (no DCGM sidecar, works anywhere CUDA is available)
- Trace export: OTLP JSON file on `$SCRATCH` (no sidecar, no ports in SLURM job)
- Grafana dashboard: primary deliverable — instrumentation exists to feed it

---

## Architecture

### Two deployment contexts

**Local (docker-compose)** — for development and dashboard validation:
```
bench_rms_norm.py
  → OTel SDK (spans + gauge metrics)
  → OTLP gRPC (4317) → otelcol
    → Prometheus exporter (8889) → Prometheus (9090) → Grafana (3000)
    → Jaeger OTLP (4317) → Jaeger UI (16686)
```

**SLURM (Narval)** — for production benchmark runs:
```
bench_rms_norm.py
  → OTel SDK
    → ConsoleSpanExporter → $SCRATCH/traces/bench_<JOBID>.jsonl
    → MLflow (existing latency + throughput metrics, unchanged)
  → pynvml → GPU samples as span events (no network required)
```

The switch is driven by `OTEL_EXPORTER_OTLP_ENDPOINT`: if set, use OTLP; otherwise,
fall back to `ConsoleSpanExporter` writing to `OTEL_SPAN_FILE` (or stdout).

### Span structure (per backend × shape)

```
span: "bench.rms_norm.<backend>"
  attributes:
    backend_name:   "cuda_oxide" | "triton" | "pytorch"
    batch_size:     int
    input_seq_len:  int  (= hidden_dim)
    kernel_variant: "rms_norm"
    kernel.name:    "rms_norm"
    sm.target:      os.environ.get("CUDA_ARCH", "sm_80")
    shape:          "2048x4096"
    latency_p50_us: float
    latency_p99_us: float
    throughput_gbs: float
  events:
    gpu.sample.start: {gpu.util_pct, gpu.mem_used_gb}
    gpu.sample.end:   {gpu.util_pct, gpu.mem_used_gb}
```

### OTel metrics (Gauges, local docker only)

| Metric | Labels |
|--------|--------|
| `kernelserve_bench_latency_p50_us` | `backend_name`, `shape`, `kernel` |
| `kernelserve_bench_latency_p99_us` | `backend_name`, `shape`, `kernel` |
| `kernelserve_bench_throughput_gbs` | `backend_name`, `shape`, `kernel` |
| `kernelserve_gpu_utilization_percent` | `backend_name`, `shape` |
| `kernelserve_gpu_memory_used_gb` | `backend_name`, `shape` |

Note: `kernelserve_kernel_latency_seconds` Histogram (defined in observability/CLAUDE.md)
is the right instrument for the streaming serving path (Phase 4). The bench harness
emits pre-aggregated gauges because it runs 1000 iterations offline and computes
percentiles; per-observation histogram recording would require restructuring the timing
loop and is out of scope.

---

## Files

### New files

| Path | Purpose |
|------|---------|
| `observability/__init__.py` | Makes `observability` importable as a package |
| `observability/otel/__init__.py` | Sub-package marker |
| `observability/metrics/__init__.py` | Sub-package marker |
| `observability/metrics/gpu.py` | pynvml sampler: `sample_gpu_metrics(device_index) -> dict` |
| `docker-compose.yml` | otelcol + Prometheus + Jaeger + Grafana for local dev |
| `slurm/bench_rms_norm_phase3.sh` | Phase 3 SLURM job (adds OTel env vars to phase2 script) |

### Modified files

| Path | Changes |
|------|---------|
| `observability/otel/instrumentation.py` | Add `create_meter()`, fix `trace_kernel` TODO (batch_size/seq_len), add file-exporter fallback |
| `observability/otel/collector.yaml` | Wire Jaeger OTLP exporter into traces pipeline |
| `experiments/bench_rms_norm.py` | Add tracer + meter init; wrap inner loop in spans; GPU sampling; emit gauge metrics |
| `observability/grafana/dashboards/kernelserve.json` | Add 5 panels + `backend` and `shape` template variables |
| `observability/prometheus/scrape_config.yaml` | Tighten scrape intervals: 5s for `kernelserve` job (request metrics) |
| `pyproject.toml` | Add `nvidia-ml-py>=12.0.0; sys_platform == "linux"` |

---

## Implementation steps

### Step 1 — Package scaffolding
Create the three `__init__.py` files. Required because `pyproject.toml` lists
`observability` as a wheel package but the directory currently lacks them, causing
`from observability.otel.instrumentation import ...` to fail.

### Step 2 — `observability/metrics/gpu.py`
```python
try:
    import pynvml
    pynvml.nvmlInit()
    _available = True
except (ImportError, Exception):
    _available = False

def sample_gpu_metrics(device_index: int = 0) -> dict[str, float]:
    if not _available:
        return {}
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu.util_pct": float(util.gpu),
        "gpu.mem_used_gb": float(mem.used) / 1024 ** 3,
    }
```
Graceful degradation: returns `{}` on macOS or when pynvml is missing so tests
pass without CUDA.

### Step 3 — `observability/otel/instrumentation.py`
Extend the existing file (do not rewrite from scratch — `create_tracer()` and
`trace_kernel()` are already correct in structure).

Changes:
- `_build_provider()`: if `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, add a
  `SimpleSpanProcessor(ConsoleSpanExporter(out=_open_span_file()))` instead of
  OTLP; `_open_span_file()` opens `OTEL_SPAN_FILE` for writing or falls back to
  `sys.stdout`.
- Add `create_meter(service_name) -> Meter`: builds a `MeterProvider` with the
  same conditional export logic (OTLP metrics → collector when endpoint is set,
  no-op otherwise). Returns `metrics.get_meter(service_name)`.
- Fix the `trace_kernel` TODO on line 54: add `batch_size` and `seq_len` as
  optional kwargs with `span.set_attribute` calls. The existing decorator is used
  by serving; don't break its call signature.

### Step 4 — `experiments/bench_rms_norm.py`
Add to `main()` preamble:
```python
from observability.otel.instrumentation import create_tracer, create_meter
from observability.metrics.gpu import sample_gpu_metrics

tracer = create_tracer("kernelserve-bench")
meter  = create_meter("kernelserve-bench")
# one Gauge per metric, created once before the loop
p50_gauge = meter.create_gauge("kernelserve_bench_latency_p50_us", unit="us")
p99_gauge = meter.create_gauge("kernelserve_bench_latency_p99_us", unit="us")
tput_gauge = meter.create_gauge("kernelserve_bench_throughput_gbs", unit="GB/s")
gpu_util_gauge = meter.create_gauge("kernelserve_gpu_utilization_percent", unit="%")
gpu_mem_gauge  = meter.create_gauge("kernelserve_gpu_memory_used_gb", unit="GB")
```

Replace the inner loop body (lines 147-153) with a span context:
```python
attrs = {"backend_name": name, "shape": f"{batch}x{hidden_dim}"}
with tracer.start_as_current_span(f"bench.rms_norm.{name}") as span:
    span.set_attribute("backend_name", name)
    span.set_attribute("batch_size", batch)
    span.set_attribute("input_seq_len", hidden_dim)
    span.set_attribute("kernel_variant", "rms_norm")
    span.set_attribute("kernel.name", "rms_norm")
    span.set_attribute("sm.target", os.environ.get("CUDA_ARCH", "sm_80"))
    span.set_attribute("shape", f"{batch}x{hidden_dim}")
    span.add_event("gpu.sample.start", attributes=sample_gpu_metrics())
    result = bencher(batch, hidden_dim)
    span.add_event("gpu.sample.end", attributes=sample_gpu_metrics())
    span.set_attribute("latency_p50_us", result.p50_us)
    span.set_attribute("latency_p99_us", result.p99_us)
    span.set_attribute("throughput_gbs", result.throughput_gbs)

log_result(result)  # MLflow unchanged

p50_gauge.set(result.p50_us, attrs | {"kernel": "rms_norm"})
p99_gauge.set(result.p99_us, attrs | {"kernel": "rms_norm"})
tput_gauge.set(result.throughput_gbs, attrs | {"kernel": "rms_norm"})
gpu_snap = sample_gpu_metrics()
if gpu_snap:
    gpu_util_gauge.set(gpu_snap["gpu.util_pct"], attrs)
    gpu_mem_gauge.set(gpu_snap["gpu.mem_used_gb"], attrs)
```

No changes to `log_result()` or `_bench_torch_fn()`.

### Step 5 — `observability/otel/collector.yaml`
Uncomment and fill the remote OTLP exporter block; add a Jaeger exporter:
```yaml
exporters:
  jaeger:
    endpoint: "${env:JAEGER_ENDPOINT:-localhost:4317}"
    tls:
      insecure: true
  # ... existing prometheus + logging exporters

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [jaeger, logging]   # was: [logging] only
```

### Step 6 — `docker-compose.yml`
```yaml
services:
  otelcol:
    image: otel/opentelemetry-collector-contrib:0.100.0
    command: ["--config=/etc/otel/config.yaml"]
    ports: ["4317:4317", "4318:4318", "8889:8889"]
    volumes: ["./observability/otel/collector.yaml:/etc/otel/config.yaml"]
    environment:
      - JAEGER_ENDPOINT=jaeger:4317

  jaeger:
    image: jaegertracing/all-in-one:1.57
    ports: ["16686:16686", "4317"]

  prometheus:
    image: prom/prometheus:v2.52.0
    ports: ["9090:9090"]
    volumes: ["./observability/prometheus/scrape_config.yaml:/etc/prometheus/prometheus.yml"]

  grafana:
    image: grafana/grafana:10.4.0
    ports: ["3000:3000"]
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    volumes:
      - "./observability/grafana/dashboards:/etc/grafana/provisioning/dashboards"
      - "./observability/grafana/provisioning:/etc/grafana/provisioning"
```

Also create `observability/grafana/provisioning/datasources/prometheus.yaml` so
Grafana auto-wires the Prometheus datasource on first boot.

### Step 7 — `observability/grafana/dashboards/kernelserve.json`
Add template variables `backend` and `shape` to the existing `templating.list`.

Add 5 panels (row layout: latency row, throughput row, GPU row):

| Panel | Type | PromQL |
|-------|------|--------|
| p50 Latency by Backend | Bar chart | `kernelserve_bench_latency_p50_us{shape=~"$shape"}` grouped by `backend_name` |
| p99 Latency by Backend | Bar chart | `kernelserve_bench_latency_p99_us{shape=~"$shape"}` grouped by `backend_name` |
| Throughput GB/s | Bar chart | `kernelserve_bench_throughput_gbs{shape=~"$shape"}` grouped by `backend_name` |
| GPU Utilization % | Time series | `kernelserve_gpu_utilization_percent` grouped by `backend_name` |
| GPU Memory Used GB | Time series | `kernelserve_gpu_memory_used_gb` grouped by `backend_name` |

All panels use the `$kernel` variable from the existing template to filter by kernel.

### Step 8 — `slurm/bench_rms_norm_phase3.sh`
Copy `bench_rms_norm_phase2.sh` verbatim, then add before the `srun` call:
```bash
export OTEL_SERVICE_NAME="kernelserve-bench"
mkdir -p "$SCRATCH/traces"
export OTEL_SPAN_FILE="$SCRATCH/traces/bench_${SLURM_JOB_ID}.jsonl"
# OTEL_EXPORTER_OTLP_ENDPOINT intentionally unset → file exporter activates
```

---

## Verification

```bash
# 1. Static checks (no GPU required)
pytest -m "not gpu"
uv run ruff check . && uv run mypy observability experiments

# 2. Local docker-compose pipeline
docker-compose up -d
MLFLOW_TRACKING_URI=file:///tmp/mlruns \
  python experiments/bench_rms_norm.py   # CPU mock, no CUDA
# → Grafana localhost:3000 — all 5 panels render (values may be 0 without GPU)
# → Jaeger localhost:16686 — 9 spans visible under service "kernelserve-bench"

# 3. SLURM (Narval)
sbatch slurm/bench_rms_norm_phase3.sh
# Expect:
#   - $SCRATCH/traces/bench_<JOBID>.jsonl contains 9 JSON span lines
#   - Each span has backend_name, batch_size, input_seq_len, kernel.name, sm.target, kernel_variant
#   - Each span has gpu.sample.start and gpu.sample.end events with util_pct + mem_used_gb
#   - MLflow still has 9 runs (unchanged from Phase 2)
```

## Definition of Done

- [ ] `pytest -m "not gpu"` passes
- [ ] `uv run ruff check . && uv run mypy observability experiments` clean
- [ ] `cargo clippy --all-targets -- -D warnings` clean (no Rust changes; trivially passes)
- [ ] SLURM job writes 9 spans to `$SCRATCH/traces/bench_<JOBID>.jsonl`
- [ ] All 6 required span attributes present (per observability/CLAUDE.md)
- [ ] `gpu.sample.start` and `gpu.sample.end` events present in each span
- [ ] `docker-compose up` + local bench run → Grafana shows all 5 panels with data
- [ ] Grafana dashboard JSON committed; no UI edits made
