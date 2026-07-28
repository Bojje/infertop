# infertop

Explain why a vLLM endpoint is slow, using only its read-only Prometheus metrics.

```console
uvx infertop diagnose http://localhost:8000
```

`infertop` polls `/metrics` twice, normalizes the result into an
engine-independent schema, and runs deterministic rules. Every finding includes
the metric evidence behind it and concrete next steps.

## Example

```text
INFERTOP
Source: http://localhost:8000/metrics
Observed: 1.0s across 2 samples

1. CRITICAL [R2_KV_THRASHING] KV cache is thrashing
   KV cache is nearly full while requests are being preempted.
   Evidence:
   - KV cache usage: 97.0% (threshold: 90.0%)
   - Preemptions: +7 over 10.0s (0.70/s)
   Remediation:
   - Reduce max model length or concurrent sequences.
   - Increase KV-cache headroom or use a smaller/quantized model.
   - Confirm improvement by checking that preemptions stop increasing.
```

For an offline fixture:

```console
uv run infertop diagnose tests/fixtures/kv_thrashing.prom \
  --previous tests/fixtures/kv_thrashing_before.prom --interval 10
```

JSON output is available with `--json`.

## Development

```console
uv sync
uv run pytest
uv run ruff check .
```

The rules are pure functions over `InferenceObservation`. Most work therefore
needs no GPU. The checked-in fixtures use vLLM's real exposition names and
shapes, but are currently hand-shaped representative snapshots; replace or
augment them with captures from the local load scenarios before publishing
benchmark claims.

## What it cannot see

- A single scrape cannot prove a counter is increasing. Live diagnosis takes
  two samples; offline diagnosis should pass `--previous`.
- Metrics explain symptoms exposed by the server, not kernel, network, client,
  model-quality, or GPU-hardware faults.
- v0.1 normalizes vLLM metrics only. It does not change server configuration,
  call admin endpoints, or retain history.
- Thresholds are intentionally conservative starting points, not universal
  capacity targets.

## Safety

Core collection is read-only: it only sends `GET` to the endpoint's `/metrics`
path and does not follow redirects.
