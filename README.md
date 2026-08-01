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

For a `/metrics` endpoint protected by bearer authentication, keep the token in an environment
variable rather than command history:

```console
export INFERTOP_API_KEY="..."
infertop diagnose https://inference.example.com
infertop watch https://inference.example.com
```

Use `--api-key-env CUSTOM_NAME` to read a different variable. The token is sent only as an
`Authorization: Bearer` header, is never included in reports, and is not forwarded through HTTP
redirects because metrics collection does not follow them.

For automation, opt into severity-based exit codes:

```console
infertop diagnose http://localhost:8000 --json --fail-on warning
```

A successful report exits `0` by default. With `--fail-on`, findings at or above the selected
`info`, `warning`, or `critical` threshold exit `1`; collection, parsing, and usage errors exit `2`.
`INCONCLUSIVE` has `info` severity, so `--fail-on info` also enforces sufficient telemetry.

## Historical Prometheus

Point `diagnose` at a Prometheus server when the live incident has already passed:

```console
infertop diagnose https://prometheus.example.com --prometheus \
  --start 2026-08-01T12:00:00Z --end 2026-08-01T12:05:00Z --step 15 \
  --prometheus-label job=vllm --prometheus-label instance=inference:8000
```

This performs one read-only `GET /api/v1/query_range` for the exact vLLM and SGLang series the
normalizers understand. Start and end accept RFC3339 values with an explicit offset or Unix
seconds. The range is hard-capped at 120 snapshots; increase `--step` for a longer window.

Use exact `--prometheus-label NAME=VALUE` filters to select one inference endpoint. If the query
returns multiple `job`/`instance` pairs, `infertop` stops instead of combining them into a false
verdict. The normal coverage and `Blocked` lines identify unavailable canonical metric families,
so historical reports remain explicit when retention or scrape configuration omitted evidence.

Bearer authentication uses the same `--api-key-env` mechanism as live collection. No token is put
in the URL or report, redirects are disabled, and `infertop` does not create recording rules,
persist results, or call the Prometheus management API.

## Live watch

The optional Textual view reruns the same diagnosis engine over a rolling sample window:

```console
uvx --from "infertop[tui]" infertop watch http://localhost:8000
```

Press `r` to refresh immediately or `q` to quit. It intentionally renders ranked findings rather
than a dashboard of unlabeled charts.

## Active API probe

`diagnose` is GET-only. When you explicitly want to execute a bounded inference request, use
`probe`:

```console
infertop probe http://localhost:8000
```

The probe discovers the first served model through `/v1/models`, sends a non-streaming
OpenAI-compatible chat completion capped at eight output tokens, and decomposes that request into
queue, TTFT/prefill, and decode using vLLM's response-level metrics. It also compares those phases
with the completion HTTP round trip. A large residual is reported as unattributed time outside the
engine phases, with network, proxy, API middleware, serialization, and client buffering as places
to investigate—not as a guessed root cause.

Repeat the exact probe sequentially when one request is too noisy:

```console
infertop probe http://localhost:8000 --count 5 --max-tokens 8
```

The repeated report prints p50/p95 for client round trip and every available server phase, plus
the number of responses that actually contained per-request metrics. It is hard-capped at 10
requests, 256 requested output tokens per request, 1,024 requested output tokens across the run,
and 32,768 prompt characters. Bounds are validated before active traffic starts. The default
remains one request and keeps the original single-request report and JSON shape.

Start vLLM with per-request metrics enabled:

```console
vllm serve Qwen/Qwen3-0.6B --enable-per-request-metrics
```

For authenticated endpoints, set `INFERTOP_API_KEY`; the CLI intentionally does not accept the
secret value as an argument.

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

With `--enable-metrics-for-all-schedulers`, SGLang labels scheduler series by `tp_rank`,
`pp_rank`, `moe_ep_rank`, and optionally `dp_rank`. `infertop` takes the busiest replicated
TP/PP/MoE value, then aggregates independent DP shards. It also prefers the explicit `priority=""`
total over per-priority breakdowns. Tokenizer request counters and latency histograms remain
workload-wide and are not multiplied by the TP degree.

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

For a local multi-GPU server, declare the exact tensor-parallel membership to enable R7:

```console
infertop diagnose http://localhost:8000 --tp-gpus 0,1
```

This performs one read-only `nvidia-smi topo -m` query. `infertop` never assumes that every GPU in
the machine belongs to the endpoint; it warns only when an explicitly declared pair has an observed
`SYS`, `NODE`, or `PHB` link. Do not use this option for a remote endpoint.

## Rules

| Rule | Evidence | Verdicts |
| --- | --- | --- |
| R1: symptom isolation | E2E, TTFT, and ITL p50/p95 | first-token-bound or all-phases-slow |
| R2: saturation | running/waiting across the sample window, KV usage | saturated, compute-bound, or queue unconfirmed |
| R3: KV health | KV usage, rising preemptions, prefix hit ratio | thrashing or low headroom |
| R4: sequence lengths | prompt/output tokens p50/p95 | prefill-bound or decode-bound |
| R5: batch efficiency | three-sample running/waiting depth, KV headroom, token rates | batch headroom or likely concurrency ceiling |
| R6: hardware correlation | sustained local GPU compute/device-memory activity plus latency | compute pressure, likely memory-bound decode, or unexplained GPU idleness |
| R7: TP topology | explicit TP GPU indices plus local `nvidia-smi topo -m` links | slow host/NUMA links inside a declared TP group |

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
  `tensor_parallel_gpu_indices` accepts an explicit local TP group and performs the same read-only
  topology diagnosis as `--tp-gpus`.
  Set `api_key_env` to the name of an environment variable when metrics require bearer auth.
- `diagnose_prometheus_range`: read-only; queries a bounded explicit historical range. Pass exact
  `labels` such as `{"job": "vllm", "instance": "inference:8000"}` to select one target, and
  keep bearer credentials in the environment named by `api_key_env`.
- `probe_inference_endpoint`: active; sends one or, with `repeat_count`, at most ten sequential
  inference POSTs. It applies the same per-request and total output ceilings as the CLI and says so
  in its tool description.

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

Rules that require three or more samples can be replayed with chronological intermediate files:

```console
uv run infertop diagnose tests/fixtures/batch_headroom.prom \
  --previous tests/fixtures/batch_headroom_before.prom \
  --intermediate tests/fixtures/batch_headroom_middle.prom \
  --interval 10
```

Repeat `--intermediate` for longer series. `--interval` is the number of seconds between adjacent
fixture files, matching the live polling option.

The checked-in fixtures use real vLLM and SGLang exposition names and shapes, but are currently
hand-shaped representative snapshots. They are not mislabeled as captures. Replace or augment them
with recorded local load scenarios before making benchmark claims.

The repository includes bounded developer tools for producing those captures. See
[Recording scenario fixtures](docs/fixture-capture.md) for the shaped traffic presets, exact raw
scrape format, provenance manifest, safety limits, and review checklist.

## Deterministic watch demo traffic

Start `infertop watch` in one terminal, then explicitly run the fixed three-stage demo in another:

```console
uv run python scripts/load_scenario.py demo-transition http://localhost:8000 \
  --confirm-active-load
```

The script prints its ceiling before sending anything, then runs paced healthy traffic, one
64-request long-output pressure burst, and paced healthy recovery: 80 requests and at most 16,896
requested output tokens in total. The stages do not accept overrides, redirects remain disabled,
and model discovery happens once.

For a small demo model, start vLLM with a deliberately low scheduler ceiling such as
`--max-num-seqs 4` so the fixed burst remains queued long enough for a three-sample watch window.
The preset creates a repeatable traffic transition; the displayed verdict still comes exclusively
from observed metrics and may differ on a server whose capacity or metric coverage is different.

## Development

```console
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

The ordered implementation plan and release gates live in [ROADMAP.md](ROADMAP.md). PyPI
publication is always a separate, manually confirmed workflow action; pushing a tag does not
publish a package.

See [Compatibility and metric shapes](docs/compatibility.md) for the tested engine/input matrix,
[SECURITY.md](SECURITY.md) for network, auth, and data-handling boundaries, and
[CONTRIBUTING.md](CONTRIBUTING.md) for fixture-first contribution requirements.

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
  long-prompt arrivals. The active probe sees at most ten sequential requests, not historical
  causality or representative concurrency.
- Engine metrics do not explain kernel, network, client, or model-quality faults. Optional NVML
  adds coarse local utilization evidence, not kernel-level profiling or hardware fault diagnosis.
- Historical Prometheus reads only the supported metric names and classic histogram series. It
  does not query exemplars, native histograms, traces, logs, or recording rules; retention gaps are
  reported as missing diagnostic coverage rather than inferred.
- SGLang's documented TP/PP/MoE/DP rank labels are normalized. Custom scheduler-rank label names
  are not yet classified and should be verified against JSON output before capacity decisions.
- Thresholds are conservative starting points, not universal SLOs or capacity targets.

## Safety

Core diagnosis only sends `GET` requests to `/metrics` or Prometheus `/api/v1/query_range` and
never follows redirects. Optional NVML collection uses device-query APIs only. These paths do not
call admin or mutation endpoints, retain history, or change server or GPU configuration. The
separately named `probe` command performs one or a small explicitly requested series of inference
`POST` calls, which consumes compute but does not alter configuration. Developer load scenarios
also require `--confirm-active-load` and enforce their ceilings before execution.

Authentication values are read from environment variables, used only for request headers, and are
not written to text or JSON reports.

## Sources

The rule flow follows Red Hat's
[5 steps to triage vLLM performance](https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance).
Metric names and aliases track the
[vLLM production metrics](https://docs.vllm.ai/en/stable/usage/metrics/) and
[per-request metrics](https://docs.vllm.ai/en/latest/features/per_request_metrics/) documentation,
plus the
[SGLang production metrics](https://docs.sglang.io/docs/references/production_metrics) reference.
Historical range collection follows the
[Prometheus HTTP API](https://prometheus.io/docs/prometheus/latest/querying/api/#range-queries).
Hardware fields follow NVIDIA's
[NVML device-query API](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) and
[utilization definitions](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html).
