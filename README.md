# infertop

Attach to a vLLM or SGLang endpoint and get a ranked, evidence-backed explanation of why inference
is slow—without Prometheus, Grafana, an agent, or GPU access.

```console
uvx infertop diagnose http://localhost:8000
```

`infertop` detects the serving engine, polls `/metrics` across a short sample window, converts
cumulative counters and histogram buckets into an engine-independent observation, and runs
deterministic rules. Every finding prints its inputs, thresholds, and engine-specific remediation.

## Example

```text
INFERTOP
Engine: vllm
Source: http://localhost:8000/metrics
Observed: 10.0s across 2 samples
Coverage: 3/5 core rules fully covered
Blocked: R1 (missing latency histograms); R5 (requires at least 3 samples)

1. CRITICAL [R3_KV_THRASHING] KV cache is thrashing
   The preemption counter is rising, proving active memory thrashing.
   Evidence:
   - KV cache usage: 97.0% (pressure threshold: 90.0%)
   - Preemptions: +7 over 10.0s (0.70/s)
   - Prefix cache hit rate: 5.0%
   Remediation:
   - Lower --max-num-seqs until preemptions stop increasing.
   - Increase cache capacity with --kv-cache-dtype fp8 where hardware supports it.
   - Use FP8/AWQ/GPTQ weights or a smaller model to leave more VRAM for KV cache.
```

Machine-readable output is available with `--json`.

## Live watch

The optional Textual view reruns the same diagnosis engine over a rolling sample window:

```console
uvx --from "infertop[tui]" infertop watch http://localhost:8000
```

Press `r` to refresh immediately or `q` to quit. It intentionally renders ranked findings rather
than a dashboard of unlabeled charts.

## Active API probe

`diagnose` is GET-only. When you explicitly want to execute one bounded inference request, use
`probe`:

```console
infertop probe http://localhost:8000
```

The probe discovers the first served model through `/v1/models`, sends a non-streaming
OpenAI-compatible chat completion capped at eight output tokens, and decomposes that request into
queue, TTFT/prefill, and decode using vLLM's response-level metrics.

Start vLLM with per-request metrics enabled:

```console
vllm serve Qwen/Qwen3-0.6B --enable-per-request-metrics
```

For authenticated endpoints, set `INFERTOP_API_KEY`; the CLI intentionally does not accept the
secret value as an argument. `--max-tokens` has a hard safety ceiling of 256.

Per-request timing is a probe result, not a workload-wide verdict. Use representative traffic and
`diagnose` for server-level conclusions.

The response-level timing object is currently a vLLM extension. SGLang endpoints still work with
`diagnose` and `watch`; `probe` falls back to explaining that phase-level response metrics were not
available.

## SGLang

Launch SGLang with Prometheus metrics enabled, then point the same commands at its server (port
30000 by default):

```console
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --enable-metrics
infertop diagnose http://localhost:30000
```

The normalizer accepts both the current `sglang:` metric namespace and the historical `sglang_`
form. Request retractions map to the canonical memory-pressure signal, and findings recommend
SGLang flags such as `--schedule-conservativeness` and `--max-running-requests`.

## Optional local GPU evidence

Explicitly opt in to read-only local NVIDIA telemetry when the endpoint is served by this machine:

```console
uvx --from "infertop[nvml]" infertop diagnose http://localhost:8000 --nvml
uvx --from "infertop[nvml,tui]" infertop watch http://localhost:8000 --nvml
```

Each metrics scrape is paired with local NVML samples for GPU compute utilization, device-memory
active time, VRAM allocation, and power draw. R6 correlates sustained hardware activity with TTFT,
ITL, and E2E latency; it does not diagnose from one instantaneous utilization value or claim that
device-memory active time measures theoretical bandwidth saturation.

`--nvml` is deliberately rejected for saved fixtures. Do not use it against a remote endpoint
unless that endpoint is actually served by the local GPUs printed in the report.

## Rules

| Rule | Evidence | Verdicts |
| --- | --- | --- |
| R1: symptom isolation | E2E, TTFT, and ITL p50/p95 | first-token-bound or all-phases-slow |
| R2: saturation | running/waiting across the sample window, KV usage | saturated, compute-bound, or queue unconfirmed |
| R3: KV health | KV usage, rising preemptions, prefix hit ratio | thrashing or low headroom |
| R4: sequence lengths | prompt/output tokens p50/p95 | prefill-bound or decode-bound |
| R5: batch efficiency | three-sample running/waiting depth, KV headroom, token rates | batch headroom or likely concurrency ceiling |
| R6: hardware correlation | sustained local GPU compute/device-memory activity plus latency | compute pressure, likely memory-bound decode, or unexplained GPU idleness |

The starting thresholds live in `Thresholds` and are printed beside evidence. Rules are pure
functions over `InferenceObservation`, so they can be tested without a GPU.

`infertop` also reports full input coverage for R1-R5. A missing metric or an empty histogram
blocks only the rules that depend on it. If R1-R3 are not fully covered, a clean threshold pass
is reported as `INCONCLUSIVE` rather than being mislabeled `HEALTHY`; the report names the
blocked rules and the exact missing or inactive evidence.

## MCP

MCP is an optional interface over the same deterministic engine:

```console
uvx --from "infertop[mcp]" infertop-mcp
```

It exposes:

- `diagnose_endpoint`: read-only; performs repeated GET scrapes of `/metrics`. Its optional
  `include_nvml` argument also reads local NVIDIA telemetry when the NVML extra is installed.
- `probe_inference_endpoint`: active; sends one bounded inference POST and says so in its tool
  description.

Example client configuration:

```json
{
  "mcpServers": {
    "infertop": {
      "command": "uvx",
      "args": ["--from", "infertop[mcp]", "infertop-mcp"]
    }
  }
}
```

## Offline fixtures

```console
uv run infertop diagnose tests/fixtures/kv_thrashing.prom \
  --previous tests/fixtures/kv_thrashing_before.prom --interval 10
```

The checked-in fixtures use real vLLM and SGLang exposition names and shapes, but are currently
hand-shaped representative snapshots. They are not mislabeled as captures. Replace or augment them
with recorded local load scenarios before making benchmark claims.

## Development

```console
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

To verify the optional MCP interface:

```console
uv sync --extra mcp
uv run --extra mcp python -c \
  "from infertop.mcp_server import create_server; print(create_server().name)"
```

To verify the optional Textual view:

```console
uv sync --extra tui
uv run --extra tui pytest tests/test_tui.py
```

To verify read-only local NVIDIA collection:

```console
uv sync --extra nvml
uv run --extra nvml infertop diagnose http://localhost:8000 --nvml
```

## What it cannot see

- One scrape cannot prove a counter is increasing or a queue is sustained. Live diagnosis takes
  three samples by default; offline diagnosis should pass `--previous`.
- Aggregate histograms cannot identify periodic ITL spikes or correlate them with individual
  long-prompt arrivals. The active probe sees one request, not historical causality.
- Engine metrics do not explain kernel, network, client, or model-quality faults. Optional NVML
  adds coarse local utilization evidence, not kernel-level profiling or hardware fault diagnosis.
- Historical Prometheus is a later slice.
- Multi-scheduler SGLang deployments can expose rank-labelled series. `infertop` supports the
  default metrics configuration; cross-rank de-duplication is not yet topology-aware.
- Thresholds are conservative starting points, not universal SLOs or capacity targets.

## Safety

Core diagnosis only sends `GET` to `/metrics` and never follows redirects. Optional NVML collection
uses device-query APIs only. Neither path calls admin or mutation endpoints, retains history, or
changes server or GPU configuration. The separately named `probe` command performs one explicitly
requested inference `POST`, which consumes compute but does not alter configuration.

## Sources

The rule flow follows Red Hat's
[5 steps to triage vLLM performance](https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance).
Metric names and aliases track the
[vLLM production metrics](https://docs.vllm.ai/en/stable/usage/metrics/) and
[per-request metrics](https://docs.vllm.ai/en/latest/features/per_request_metrics/) documentation,
plus the
[SGLang production metrics](https://docs.sglang.io/docs/references/production_metrics) reference.
Hardware fields follow NVIDIA's
[NVML device-query API](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) and
[utilization definitions](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html).
