# Security policy

## Supported code

Before the first release, security fixes target the current `main` branch. A version support table
will be added when published versions exist. Do not infer support from the historical `v0.1.0`
tag; it was never published.

## Report a vulnerability

Do not put credentials, private endpoints, captured metrics, prompts, or exploit details in a
public issue. Use GitHub's private vulnerability-reporting form from the repository Security tab
when it is available. If it is unavailable, open a minimal issue asking the maintainer to establish
a private channel, without including vulnerability details. Ordinary bugs without sensitive data
can use the public issue tracker.

Include the affected commit or version, the command/interface involved, the smallest safe
reproduction, and the impact. Redact bearer tokens, URL credentials, tenant labels, model prompts,
and customer metric values.

## Network and mutation boundaries

`infertop` has a deliberately small core boundary:

| Interface | Network or host operation | Mutates the server? |
| --- | --- | --- |
| `diagnose` / `watch` | repeated `GET` requests to `/metrics` | No |
| `diagnose --prometheus` | one `GET` to `/api/v1/query_range` | No |
| `--nvml` | local NVML device queries | No |
| `--tp-gpus` | one local `nvidia-smi topo -m` subprocess | No |
| `probe` | `GET /v1/models`, then 1–10 `POST /v1/chat/completions` | Consumes inference compute |
| developer load scenarios | model discovery plus bounded chat-completion `POST` requests | Consumes inference compute |

Core collectors never call admin, cache-flush, model-update, weight-update, or other control-plane
endpoints. They do not create Prometheus recording rules or persist a history. HTTP redirects are
disabled everywhere, so an authorization header is not forwarded to a redirected host.

The active interfaces are separately named and opt-in. `probe` enforces 10 requests, 256 requested
output tokens per request, 1,024 requested output tokens per series, and 32,768 prompt characters.
Developer scenarios require `--confirm-active-load` and validate their request, concurrency,
prompt, output, and timeout limits before execution.

## Authentication and URLs

Put bearer values in `INFERTOP_API_KEY` or another environment variable selected with
`--api-key-env`. The CLI and MCP tools accept the variable name, never a secret-valued argument.
Values are used only in an `Authorization: Bearer` header and are not included in text or JSON
reports.

Live metrics collection redacts URL userinfo, query parameters, and fragments from displayed
sources and error messages. Query parameters supplied to a metrics URL are still sent to that
exact endpoint, so do not use them for credentials. The active OpenAI-compatible probe rejects URL
userinfo and discards URL query parameters. Historical Prometheus input rejects both URL userinfo
and query parameters; use label filters for target selection and environment-based bearer auth.

Environment variables are inherited by the optional MCP stdio server. Treat the MCP client
configuration and its process environment as secret-bearing configuration. Use a narrowly scoped
read token for `/metrics` or Prometheus whenever the upstream service supports one; do not reuse an
admin token.

## Transport and output data

Use HTTPS for remote endpoints. `infertop` relies on the host's CA trust configuration and does not
offer an insecure TLS bypass. It does not pin certificates.

Reports contain endpoint host/path, engine type, aggregate performance metrics, finding evidence,
and optional local GPU names/UUIDs. Fixture manifests add model, engine version, server command,
timestamps, and content hashes. Review all of those fields before sharing a report or committing a
capture. Raw `.prom` fixture files are preserved byte-for-byte and may contain custom labels; the
capture tool cannot know which label values are sensitive.

Prompts and completion text are not printed by `probe` or the load generator, but they are sent to
the selected inference endpoint. Shell history can retain a prompt passed with `--prompt`; prefer a
non-sensitive probe string.

## Local GPU scope

`--nvml` and `--tp-gpus` inspect the machine running `infertop`, not the remote endpoint. Use them
only when that machine hosts the endpoint. Tensor-parallel membership must be declared explicitly;
the tool does not assume every installed GPU belongs to a server.
