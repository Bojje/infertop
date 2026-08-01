"""Optional MCP interface over infertop's deterministic engine."""

from __future__ import annotations

import json
import os
from typing import Any

from infertop.collector import collect_endpoint
from infertop.probe import probe_endpoint
from infertop.prometheus_api import collect_prometheus_range, parse_range_time
from infertop.report import render_json
from infertop.rules import diagnose


def diagnose_endpoint_result(
    endpoint: str,
    *,
    interval_seconds: float = 5.0,
    timeout_seconds: float = 5.0,
    sample_count: int = 3,
    include_nvml: bool = False,
    api_key_env: str = "INFERTOP_API_KEY",
    tensor_parallel_gpu_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Return structured deterministic findings from read-only metrics scrapes."""

    observation = collect_endpoint(
        endpoint,
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
        sample_count=sample_count,
        include_nvml=include_nvml,
        api_key=os.environ.get(api_key_env),
        tensor_parallel_gpu_indices=tuple(tensor_parallel_gpu_indices or ()),
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


def diagnose_prometheus_range_result(
    endpoint: str,
    *,
    start: str,
    end: str,
    step_seconds: float = 15.0,
    timeout_seconds: float = 5.0,
    labels: dict[str, str] | None = None,
    api_key_env: str = "INFERTOP_API_KEY",
) -> dict[str, Any]:
    """Return deterministic findings from a read-only Prometheus range query."""

    observation = collect_prometheus_range(
        endpoint,
        start=parse_range_time(start),
        end=parse_range_time(end),
        step_seconds=step_seconds,
        timeout_seconds=timeout_seconds,
        labels=labels,
        api_key=os.environ.get(api_key_env),
    )
    return json.loads(render_json(observation, diagnose(observation)))


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
            "Metrics and Prometheus diagnosis tools are read-only. "
            "probe_inference_endpoint sends one bounded request."
        ),
    )

    @server.tool()
    def diagnose_endpoint(
        endpoint: str,
        interval_seconds: float = 5.0,
        timeout_seconds: float = 5.0,
        sample_count: int = 3,
        include_nvml: bool = False,
        api_key_env: str = "INFERTOP_API_KEY",
        tensor_parallel_gpu_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Read metrics; optionally use env auth, local NVML, and an explicit TP topology."""

        return diagnose_endpoint_result(
            endpoint,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            sample_count=sample_count,
            include_nvml=include_nvml,
            api_key_env=api_key_env,
            tensor_parallel_gpu_indices=tensor_parallel_gpu_indices,
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

    @server.tool()
    def diagnose_prometheus_range(
        endpoint: str,
        start: str,
        end: str,
        step_seconds: float = 15.0,
        timeout_seconds: float = 5.0,
        labels: dict[str, str] | None = None,
        api_key_env: str = "INFERTOP_API_KEY",
    ) -> dict[str, Any]:
        """Read a bounded Prometheus range; use labels to select one inference endpoint."""

        return diagnose_prometheus_range_result(
            endpoint,
            start=start,
            end=end,
            step_seconds=step_seconds,
            timeout_seconds=timeout_seconds,
            labels=labels,
            api_key_env=api_key_env,
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
