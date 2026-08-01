# Changelog

All notable user-facing changes are recorded here. This project follows semantic versioning while
its output schemas and thresholds remain explicitly alpha.

## 0.2.0 - Unreleased

First intended public package version. The historical `v0.1.0` Git tag was never published and is
not reused.

### Diagnosis engine

- Normalize vLLM and SGLang exposition into one engine-independent observation schema.
- Diagnose symptom phase, saturation, KV-cache health, request-length shape, batching headroom,
  sustained local GPU correlation, and explicitly declared tensor-parallel topology (R1–R7).
- Rank every finding with observed evidence, thresholds, missing coverage, and engine-specific
  remediation; emit text or schema-versioned JSON and optional severity exit codes.
- Handle counter resets, chronological multi-snapshot windows, classic histogram deltas/quantiles,
  SGLang historical names, priority totals, and scheduler TP/PP/MoE/DP rank shapes.

### Inputs and interfaces

- Diagnose repeated live `/metrics` GETs, saved exposition series, or bounded Prometheus
  `/api/v1/query_range` results with exact label filters and bearer authentication.
- Add optional Textual watch mode, local NVML telemetry, explicit `nvidia-smi topo -m` evidence,
  and MCP tools over the same deterministic engine.
- Add an explicitly active OpenAI-compatible probe with response-phase correlation, optional
  1–10-request p50/p95 summaries, and hard request/prompt/output ceilings.

### Development and distribution

- Add bounded healthy, queue, KV, prefill, decode, and fixed healthy-pressure-recovery traffic
  scenarios plus provenance-aware raw fixture capture tooling.
- Test Python 3.11–3.13, optional interfaces, distribution metadata, wheel build, and clean core
  installation in CI.
- Keep releases manually dispatched and require an existing exact version tag; PyPI publication is
  a separate boolean confirmation protected by the `pypi` environment.
- Document compatibility, security/data boundaries, fixture provenance, contribution rules, and
  honest limitations.
