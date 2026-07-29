# infertop

Attach to a vLLM endpoint and get a ranked, evidence-backed explanation of why inference is
slow—without Prometheus, Grafana, an agent, or GPU access.

```console
uvx infertop diagnose http://localhost:8000
```

`infertop` polls `/metrics` twice, converts cumulative counters and histogram buckets into a
windowed engine-independent observation, and runs deterministic rules. Every finding prints its
inputs, thresholds, and concrete remediation.

## Example

```text
INFERTOP
Source: http://localhost:8000/metrics
Observed: 5.0s across 2 samples

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

## Rules

| Rule | Evidence | Verdicts |
| --- | --- | --- |
| R1: symptom isolation | E2E, TTFT, and ITL p50/p95 | first-token-bound or all-phases-slow |
| R2: saturation | running/waiting across two scrapes, KV usage | saturated, compute-bound, or queue unconfirmed |
| R3: KV health | KV usage, rising preemptions, prefix hit ratio | thrashing or low headroom |
| R4: sequence lengths | prompt/output tokens p50/p95 | prefill-bound or decode-bound |

The starting thresholds live in `Thresholds` and are printed beside evidence. Rules are pure
functions over `InferenceObservation`, so they can be tested without a GPU.

## MCP

MCP is an optional interface over the same deterministic engine:

```console
uvx --from "infertop[mcp]" infertop-mcp
```

It exposes:

- `diagnose_endpoint`: read-only; performs two GET scrapes of `/metrics`.
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

The checked-in fixtures use real vLLM exposition names and shapes, but are currently hand-shaped
representative snapshots. They are not mislabeled as captures. Replace or augment them with
recorded local load scenarios before making benchmark claims.

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

## What it cannot see

- One scrape cannot prove a counter is increasing or a queue is sustained. Live diagnosis takes
  two samples; offline diagnosis should pass `--previous`.
- Aggregate histograms cannot identify periodic ITL spikes or correlate them with individual
  long-prompt arrivals. The active probe sees one request, not historical causality.
- Metrics explain symptoms exposed by the server, not kernel, network, client, model-quality, or
  GPU-hardware faults.
- v0.1 normalizes vLLM only. SGLang normalization, R5 batch efficiency, NVML fusion, and historical
  Prometheus are later slices.
- Thresholds are conservative starting points, not universal SLOs or capacity targets.

## Safety

Core diagnosis only sends `GET` to `/metrics` and never follows redirects. It does not call admin
or mutation endpoints, retain history, or change server configuration. The separately named
`probe` command performs one explicitly requested inference `POST`, which consumes compute but
does not alter configuration.

## Sources

The rule flow follows Red Hat's
[5 steps to triage vLLM performance](https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance).
Metric names and aliases track the
[vLLM production metrics](https://docs.vllm.ai/en/stable/usage/metrics/) and
[per-request metrics](https://docs.vllm.ai/en/latest/features/per_request_metrics/) documentation.
