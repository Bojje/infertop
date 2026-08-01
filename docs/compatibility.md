# Compatibility and metric shapes

`infertop` supports observed metric shapes, not engine marketing version ranges. Engines can add,
remove, or relabel metrics independently of their API compatibility. The report's `Coverage` and
`Blocked` fields are the runtime authority: missing evidence makes an affected rule inconclusive
instead of silently falling back to a guess.

## Runtime support

| Surface | Tested in CI | Notes |
| --- | --- | --- |
| Python | 3.11, 3.12, 3.13 on Linux | Core requires Python 3.11 or newer. |
| Core wheel | clean install without extras | Only `httpx` is a core runtime dependency. |
| Textual | optional `tui` extra | The TUI calls the same collector and rule engine as report mode. |
| NVIDIA telemetry | optional `nvml` extra | Unit-tested with synthetic devices; requires local NVIDIA driver access. |
| MCP | optional `mcp` extra | Stdio wrapper over the same diagnosis and probe functions. |
| Windows | not native-CI tested | Remote HTTP diagnosis should be portable; WSL2 behaves as Linux. |

On 2026-08-01, the locally built wheel was clean-installed and executed independently through
`uvx --isolated`, standard `pip` in a fresh virtualenv, and `pipx` in a fresh tool home. None of
those checks installed the TUI, NVML, or MCP extras. CI repeats the wheel build and clean core
installation on every change.

The current local development host has an RTX 5080 under WSL2 and has exercised read-only
`nvidia-smi` discovery/topology. That is a smoke check, not a supported-GPU certification matrix.

## Engine shapes

| Engine shape | Normalization status | Evidence in this repository |
| --- | --- | --- |
| vLLM `vllm:` namespace | Supported | Synthetic single-rank scenario fixtures and golden findings. |
| vLLM aliases such as `gpu_cache_usage_perc` and prefix counters with/without `_total` | Supported | Normalizer unit fixtures. |
| SGLang current `sglang:` namespace | Supported | Synthetic healthy and pressure fixtures. |
| SGLang historical `sglang_` namespace | Supported | Historical-name fixture regression. |
| SGLang scheduler rank labels | Supported for known `tp_rank`, `pp_rank`, `moe_ep_rank`, `rank`, and `dp_rank` shapes | Multi-rank gauges, counters, priority totals, and histogram regressions. |
| Unknown namespaces or custom rank-label layouts | Not normalized | Engine detection fails, or coverage reports missing families. |

No checked-in scenario is currently claimed as a real vLLM/SGLang capture. The synthetic fixtures
use documented exposition names and are intentionally labelled representative. Live captures with
engine-version provenance remain a release-readiness task.

The current-name rows track the official
[vLLM metrics reference](https://docs.vllm.ai/en/stable/design/metrics/) and
[SGLang production metrics reference](https://docs.sglang.io/docs/references/production_metrics).

## Canonical metric families

| Canonical evidence | vLLM names | SGLang names | Used by |
| --- | --- | --- | --- |
| running / waiting | `num_requests_running`, `num_requests_waiting` | `num_running_reqs`, `num_queue_reqs` | R2, R5 |
| KV usage | `kv_cache_usage_perc`, historical `gpu_cache_usage_perc` | `token_usage` | R2, R3, R5 |
| preemptions / retractions | `num_preemptions_total` | `num_retracted_requests_total` | R3 |
| prefix reuse | prefix query/hit counters | `cache_hit_rate` gauge | R3 |
| token throughput | prompt/generation token counters | prompt/generation token counters | R5 |
| E2E / queue / TTFT / TPOT | classic Prometheus histograms | classic Prometheus histograms | R1, R2, R6 |
| prompt / output lengths | request token histograms | token histograms | R4 |
| prefill / decode duration | request phase histograms where emitted | not currently mapped | Normalized, not yet used by a rule |

Histogram support means the classic `_bucket`, `_count`, and optional `_sum` series. Native
Prometheus histograms and summaries are not normalized. SGLang prompt/generation token histograms
require its token-histogram collection option; if absent, only R4 is blocked.

SGLang's all-schedulers mode emits scheduler-local data from multiple TP ranks. `infertop`
de-duplicates replicated TP/PP/MoE values by scheduler group, preserves the busiest replica, and
sums independent DP groups. It prefers an explicit `priority=""` total over per-priority
breakdowns. Custom rank labels are not guessed.

The mode itself is documented in SGLang's
[server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) as recording
metrics separately on every scheduler/TP rank.

## Input adapters

| Input | Support | Limits |
| --- | --- | --- |
| Live `/metrics` | vLLM and SGLang exposition text | 2 or more samples; 3 by default; no redirects. |
| Saved `.prom` series | same parser and normalizer | Caller supplies chronological files and interval. |
| Prometheus HTTP API | matrix result from `/api/v1/query_range` | Exact supported-name selector; maximum 120 timestamps; one job/instance target. |
| Local NVML | GPU utilization, device-memory activity, allocation, power | Correlation only; no kernel counters or remote-host inference. |
| `nvidia-smi topo -m` | explicit local TP group | Warns only for observed `SYS`, `NODE`, or `PHB` pairs. |

Prometheus range input supports classic float sample matrices. It does not consume exemplars,
native histograms, traces, logs, remote-read protobufs, or a Grafana data source.

## Active probe compatibility

The active probe uses non-streaming OpenAI-compatible `/v1/models` and
`/v1/chat/completions`. Basic completion success and token usage work with compatible servers.
Phase decomposition additionally expects vLLM's response-level `metrics` object containing
`time_to_first_token_ms`, `generation_time_ms`, `queue_time_ms`, `mean_itl_ms`, and
`tokens_per_second`, enabled with `--enable-per-request-metrics`.

Those fields and their single-request semantics follow vLLM's
[per-request metrics reference](https://docs.vllm.ai/en/stable/features/per_request_metrics/).

SGLang currently falls back to a successful request report without phase-level response metrics.
That is not treated as an engine diagnosis. Repeated probe percentiles are sequential client
measurements and must not be interpreted as concurrent workload distributions.

## Adding or changing support

Metric compatibility changes require an exposition fixture, a normalizer assertion, and a golden
top-finding test when the shape affects a verdict. A version number without the corresponding raw
metric evidence is not sufficient. See [Recording scenario fixtures](fixture-capture.md) and
[CONTRIBUTING.md](../CONTRIBUTING.md).
