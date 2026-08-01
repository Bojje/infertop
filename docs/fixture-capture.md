# Recording scenario fixtures

The checked-in synthetic fixtures make rule development independent of a GPU. Recorded fixtures
add credibility and version coverage; they do not replace synthetic edge cases.

## 1. Start a small server

For example, on a CUDA-capable vLLM host:

```console
vllm serve Qwen/Qwen3-0.6B --enable-per-request-metrics
```

Record the exact engine version, model identifier, and server command. Do not put API keys or other
secrets in the recorded command.

## 2. Start chronological capture

The output path must be new, which prevents accidentally overwriting an existing fixture:

```console
uv run python scripts/capture_metrics.py http://localhost:8000 \
  tests/fixtures/recorded/vllm-healthy-0.10.1 \
  --scenario healthy --samples 20 --interval 1 \
  --engine vllm --engine-version 0.10.1 \
  --model Qwen/Qwen3-0.6B \
  --server-command "vllm serve Qwen/Qwen3-0.6B --enable-per-request-metrics"
```

Capture preserves each decoded HTTP response body byte-for-byte as `000.prom`, `001.prom`, and so
on. A `manifest.json` records timestamps, hashes, source without its query string, and the supplied
provenance. Bearer values are read from `INFERTOP_API_KEY` (or the environment variable selected by
`--api-key-env`) and are never written to the manifest.

## 3. Run one shaped workload

In another terminal, inspect the preset first:

```console
uv run python scripts/load_scenario.py --help
```

Then explicitly confirm the active inference load:

```console
uv run python scripts/load_scenario.py healthy http://localhost:8000 \
  --confirm-active-load
```

Available presets are `healthy`, `queue-saturated`, `kv-pressure`, `prefill-bound`, and
`decode-bound`. Each has hard limits of 128 requests, concurrency 32, 8,192 approximate prompt
words, and 256 output tokens per request. Overrides beyond those limits are rejected before any
request is sent. The per-request marker occurs in the first prompt block so prefix caching does not
silently convert long-prompt scenarios into cache-hit measurements.

The prompt asks for the intended output length, and the load summary prints the server-reported
prompt and completion token totals. Review those totals before accepting a capture: a token ceiling
does not by itself prove that the model generated a long response. To make queue saturation
repeatable on a small model, start a separate capture server with a deliberately low scheduler
limit such as `--max-num-seqs 4`; record that flag in the manifest rather than claiming the traffic
preset guarantees a queue on every server.

Run only one scenario during a capture. Allow an idle baseline at the beginning and end. A scenario
name describes traffic shape, not a guaranteed verdict: the golden expectation must be based on
the captured evidence and the tested hardware/server configuration.

For a live TUI recording rather than a single fixture shape, `demo-transition` is a fixed
healthy-baseline, queue-pressure, healthy-recovery sequence:

```console
uv run python scripts/load_scenario.py demo-transition http://localhost:8000 \
  --confirm-active-load
```

It prints the 80-request and 16,896-requested-output-token ceiling before traffic, paces the first
and last eight requests at one launch per second, and sends a 64-request/concurrency-16 pressure
burst between them. Overrides are rejected so recordings use the same traffic definition. For a
small model, use a recorded server flag such as `--max-num-seqs 4` to keep the pressure stage
visible across the watch window; do not claim a saturation verdict unless the captured metrics
show it.

## 4. Review before committing

- Confirm every `.prom` file parses with `infertop diagnose` in chronological order.
- Inspect the manifest for hostnames or command arguments that should not be public.
- Keep the smallest consecutive series that demonstrates the finding.
- Add a golden test asserting the top finding, engine version, and fixture provenance.
- Label hand-shaped fixtures as synthetic; never call them measured benchmarks.
