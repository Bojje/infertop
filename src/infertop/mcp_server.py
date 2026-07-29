"""Optional MCP interface over infertop's deterministic engine."""

from __future__ import annotations

import json
import os
from typing import Any

from infertop.collector import collect_endpoint
from infertop.probe import probe_endpoint
from infertop.report import render_json
from infertop.rules import diagnose


def diagnose_endpoint_result(
    endpoint: str,
    *,
    interval_seconds: float = 5.0,
    timeout_seconds: float = 5.0,
    sample_count: int = 3,
) -> dict[str, Any]:
    """Return structured deterministic findings from read-only metrics scrapes."""

    observation = collect_endpoint(
        endpoint,
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
        sample_count=sample_count,
    )
    return json.loads(render_json(observation, diagnose(observation)))


def probe_inference_endpoint_result(
    endpoint: str,
    *,
    model: str | None = None,
    prompt: str = "Reply with exactly the word OK.",
    max_tokens: int = 8,
    api_key_env: str = "INFERTOP_API_KEY",
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Send one bounded generation request and return its per-request metrics."""

    return probe_endpoint(
        endpoint,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        api_key=os.environ.get(api_key_env),
        timeout_seconds=timeout_seconds,
    ).to_dict()


def create_server() -> Any:
    """Create the optional FastMCP server without loading MCP in core installs."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError('MCP support requires: pip install "infertop[mcp]"') from exc

    server = FastMCP(
        "infertop",
        instructions=(
            "Diagnose vLLM and SGLang endpoints with deterministic, evidence-backed rules. "
            "diagnose_endpoint is read-only. probe_inference_endpoint sends one bounded request."
        ),
    )

    @server.tool()
    def diagnose_endpoint(
        endpoint: str,
        interval_seconds: float = 5.0,
        timeout_seconds: float = 5.0,
        sample_count: int = 3,
    ) -> dict[str, Any]:
        """Read /metrics repeatedly and return ranked, evidence-backed findings."""

        return diagnose_endpoint_result(
            endpoint,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            sample_count=sample_count,
        )

    @server.tool()
    def probe_inference_endpoint(
        endpoint: str,
        model: str | None = None,
        prompt: str = "Reply with exactly the word OK.",
        max_tokens: int = 8,
        api_key_env: str = "INFERTOP_API_KEY",
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Opt in to one bounded inference POST and return per-request timing evidence."""

        return probe_inference_endpoint_result(
            endpoint,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
        )

    return server


def main() -> None:
    """Run the MCP server over stdio."""

    try:
        create_server().run(transport="stdio")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
