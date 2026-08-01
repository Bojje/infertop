# Contributing

Thanks for helping make inference diagnosis less mystical. The core design constraint is simple:
normalize evidence first, then run pure deterministic rules. UI and protocol integrations must
call that same engine rather than implement a second diagnosis path.

## Development setup

```console
git clone https://github.com/Bojje/infertop.git
cd infertop
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

CI runs the core suite on Python 3.11–3.13 and separately constructs the MCP server, Textual TUI,
NVML integration, and clean wheel. Keep core dependencies small; engine SDKs, CUDA frameworks, and
storage clients do not belong in the default install.

## Rules and normalization

Rules accept `InferenceObservation` and return findings without network, filesystem, environment,
or GPU access. Every non-healthy finding needs:

- a stable rule identifier and severity;
- the observed values and thresholds in `evidence`;
- a conclusion no stronger than those values support;
- concrete remediation, with engine-specific flags only when the engine is known;
- fixture-first tests, including the no-verdict case for missing evidence.

Engine-specific names belong in the normalizer. Do not reference raw vLLM or SGLang metric names
inside a rule. New aliases need a minimal raw exposition fixture and an assertion for the canonical
field. Rank-labelled metrics also need tests proving replicated values are not double-counted and
independent shards are not dropped.

## Fixtures and golden reports

Synthetic fixtures are appropriate for edge cases and fast deterministic CI. Label them as
synthetic; do not describe hand-shaped values as benchmark results or real captures. A scenario
golden file should assert the highest-ranked finding while focused unit tests cover the evidence
details and thresholds.

Real captures must use the bounded capture workflow in
[docs/fixture-capture.md](docs/fixture-capture.md). Review raw metric labels, source hostnames,
model identifiers, server commands, and manifests for sensitive values before committing. Keep the
smallest chronological window that demonstrates the behavior and record engine version/model/load
provenance.

## Safety and interfaces

Core collection stays read-only: only metrics and historical query GETs, local query APIs, and the
read-only topology command are allowed. A feature that sends inference must be explicitly named as
active, opt-in, bounded before execution, and documented with its maximum request/token cost.
Never add an admin/control endpoint, automatic tuning action, or redirect-following behavior.

Secrets come from environment variables, not CLI values, reports, fixtures, exception strings, or
tests. Add a non-leak regression whenever touching URLs, authentication, capture manifests, or HTTP
errors. Read [SECURITY.md](SECURITY.md) before changing a network boundary.

Optional TUI, MCP, NVML, Prometheus, and future engine adapters are interfaces over the canonical
schema. They should not fork thresholds or verdict logic.

## Pull request checklist

- The change is scoped and its user-visible behavior is documented.
- Ruff formatting/lint and pytest pass locally.
- New metric shapes and findings have fixtures and negative tests.
- Findings print evidence, thresholds, and remediation.
- Active traffic and external writes have explicit bounds/confirmation.
- No secret, private endpoint, customer label, or misleading benchmark claim is present.
- The core wheel still installs without optional extras.

Maintainers handle versioning, tags, releases, and PyPI publication separately from ordinary pull
requests. A merged change never implies that a package should be published.
