# Roadmap

`infertop` is developed diagnosis-engine first. The report is the product; the Textual interface
is a demonstration of the same deterministic engine. Work below is ordered so missing GPU or
serving-engine setup never blocks pure rule development.

## Current baseline

The repository already provides vLLM and SGLang normalization, multi-snapshot observations,
rules R1-R6, text and JSON reports, a live Textual view, optional local NVML evidence, a bounded
OpenAI-compatible request probe, MCP tools, bearer authentication, and severity-based exit codes.

The checked-in Prometheus fixtures are representative hand-shaped inputs. They exercise the rule
engine credibly, but they are not presented as real captures. Historical Prometheus queries,
multi-rank SGLang aggregation, and tensor-parallel topology diagnosis remain intentionally
unsupported until the milestones below land.

## Milestone 1: release safety

- Continuously build, validate, and clean-install the wheel in CI.
- Keep releases manual: require an existing exact version tag and make PyPI publication a separate
  boolean confirmation guarded by the `pypi` environment.
- Never publish from a tag push alone.
- Before the first publication, recheck name availability, choose a fresh version (the historical
  `v0.1.0` tag cannot be reused), and configure PyPI trusted publishing for the exact repository,
  workflow, and environment.

Exit criterion: a pull request proves the distributable works, while no normal push or tag can
publish it.

## Milestone 2: recorded scenario fixtures

Status: bounded load generation, raw capture, provenance manifests, and fake-endpoint tests are in
place. Captures from a real vLLM process and their golden findings are still pending.

- [x] Add a small bounded async load generator for healthy traffic, burst overload, long-prompt/short-
  output prefill pressure, short-prompt/long-output decode pressure, and KV pressure.
- [x] Add a capture utility that records chronological `/metrics` scrapes plus engine, engine version,
  model, command, interval, and scenario provenance.
- [ ] Run the scenarios against a small vLLM model such as Qwen3-0.6B on the local RTX 5080 when the
  environment is ready.
- [ ] Check in compact real captures and golden top-finding tests. Keep synthetic fixtures in an
  explicitly labelled directory for edge cases that are difficult or unsafe to induce.

Exit criterion: each primary diagnosis has at least one recorded fixture and a golden assertion,
with capture provenance that another contributor can reproduce.

## Milestone 3: R7 tensor-parallel topology

Status: the canonical topology schema, pure matrix parser, local one-GPU fixture, and synthetic
multi-GPU link fixtures are implemented. Explicit tensor-parallel membership and R7 findings remain.

- [x] Parse `nvidia-smi topo -m` into an engine-independent topology model using a pure parser.
- Collect topology only when the user explicitly opts into local NVIDIA evidence.
- Warn conservatively when a multi-GPU tensor-parallel group crosses known slow paths such as
  `SYS`, `NODE`, or `PHB`; print the exact GPU pair, link class, and legend as evidence.
- Treat a single GPU, unknown mappings, and incomplete topology as no verdict rather than guessing.
- Test with synthetic one-GPU, NVLink, PCIe-switch, and cross-NUMA fixtures; the local one-GPU RTX
  5080 is only a no-op live check.

Exit criterion: R7 is deterministic, fixture-tested, read-only, and cannot make a topology claim
without an observed GPU pair and link class.

## Milestone 4: SGLang multi-rank correctness

- Add rank-labelled current and historical SGLang fixtures.
- Distinguish replicated counters/histograms from truly sharded gauges.
- De-duplicate replicated values and aggregate shard-local values without hiding per-rank pressure.
- Document the supported SGLang metric/version shapes and report inconclusive coverage for unknown
  layouts.

Exit criterion: equivalent single-rank and multi-rank workloads normalize to equivalent canonical
rates and quantiles, with regression tests preventing double counting.

## Milestone 5: historical Prometheus input

- Add a read-only Prometheus HTTP API adapter using the existing `httpx` dependency.
- Support an explicit time range and step, bearer credentials from an environment variable, and
  the same canonical observation consumed by live and fixture sources.
- Use mocked API payloads and golden reports in CI; do not add storage, recording rules, or a
  dashboard layer.
- Make query coverage and unavailable metric families visible in reports.

Exit criterion: a saved fixture, a live engine, and a Prometheus range can drive the same pure
rules with equivalent evidence semantics.

## Milestone 6: request drill-down and deterministic demo

- Extend the explicitly active probe with a small hard-capped repeat count and p50/p95 summaries,
  while keeping request count and token ceilings obvious.
- Add a staged load scenario that transitions from healthy to saturated or thrashing and back.
- Use that scenario to make Textual verdict transitions deterministic, then record the short README
  GIF. Do not turn the TUI into a long-term dashboard.

Exit criterion: the demo visibly reflects the same findings as report mode, and active traffic is
always bounded and opt-in.

## Milestone 7: documentation and compatibility

- Put a real captured report and 30-second demo in the README.
- Publish an engine/version/metric compatibility matrix, authentication and security guidance,
  fixture capture instructions, and an honest limitations section.
- Keep every output finding tied to observed evidence, thresholds, and actionable remediation.
- Verify core installation with `uvx`, `pip`, and `pipx` without optional TUI, NVML, or MCP
  dependencies.

Exit criterion: a new user can install, diagnose, interpret limitations, and reproduce fixtures
without reading source code.

## Milestone 8: explicit release and launch

- With separate maintainer approval, configure the PyPI trusted publisher and manually run the
  verified release workflow for a fresh tag.
- Confirm the public artifact from a clean environment before creating launch posts.
- Launch to Show HN, r/LocalLLaMA, the vLLM community, and later the MCP registry and relevant
  ecosystem lists. Reference the Red Hat triage runbook that motivated the rule flow.

Exit criterion: the published package matches the tested tag and wheel, installation commands are
verified, and launch claims use recorded evidence rather than synthetic benchmark claims.

## Non-goals and permanent guardrails

- Core diagnosis remains read-only and never calls admin or mutation endpoints.
- Active inference stays a separately named, bounded, opt-in probe.
- No long-term storage, hosted control plane, or Grafana replacement.
- No black-box verdicts: every rule cites its inputs, thresholds, and missing coverage.
- Optional integrations remain interfaces over the canonical engine, not alternate diagnosis
  implementations.
