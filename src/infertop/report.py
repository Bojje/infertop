"""Human and machine-readable reports."""

from __future__ import annotations

import json
from dataclasses import asdict

from infertop.probe import ProbeResult
from infertop.rules import Finding
from infertop.schema import InferenceObservation


def _gpu_summary(observation: InferenceObservation) -> list[str]:
    lines = []
    for gpu in observation.current.gpus:
        fields = []
        if gpu.gpu_utilization is not None:
            fields.append(f"compute {gpu.gpu_utilization:.0%}")
        if gpu.memory_io_utilization is not None:
            fields.append(f"memory active {gpu.memory_io_utilization:.0%}")
        if gpu.vram_usage is not None:
            fields.append(f"VRAM {gpu.vram_usage:.0%}")
        if gpu.power_watts is not None and gpu.power_limit_watts is not None:
            fields.append(f"power {gpu.power_watts:.0f}/{gpu.power_limit_watts:.0f}W")
        details = ", ".join(fields) if fields else "telemetry unavailable"
        lines.append(f"GPU {gpu.index}: {gpu.name} ({details})")
    return lines


def render_text(observation: InferenceObservation, findings: tuple[Finding, ...]) -> str:
    interval = observation.interval_seconds
    sample_summary = (
        f"{interval:.1f}s across {observation.sample_count} samples"
        if interval is not None
        else "single snapshot"
    )
    lines = [
        "INFERTOP",
        f"Engine: {observation.current.engine}",
        f"Source: {observation.current.source}",
        f"Observed: {sample_summary}",
    ]
    if observation.current.gpus:
        lines.extend(("Hardware: local NVML", *_gpu_summary(observation)))
    lines.append("")
    for index, finding in enumerate(findings, 1):
        lines.extend(
            [
                f"{index}. {finding.severity.upper()} [{finding.rule_id}] {finding.title}",
                f"   {finding.summary}",
                "   Evidence:",
                *(f"   - {item}" for item in finding.evidence),
                "   Remediation:",
                *(f"   - {item}" for item in finding.remediations),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def render_json(observation: InferenceObservation, findings: tuple[Finding, ...]) -> str:
    gpus = [
        {
            **asdict(gpu),
            "vram_usage": gpu.vram_usage,
            "power_ratio": gpu.power_ratio,
        }
        for gpu in observation.current.gpus
    ]
    payload = {
        "schema_version": 1,
        "engine": observation.current.engine,
        "source": observation.current.source,
        "sample_count": observation.sample_count,
        "interval_seconds": observation.interval_seconds,
        "hardware": {"source": "local_nvml", "gpus": gpus} if gpus else None,
        "findings": [asdict(finding) for finding in findings],
    }
    return json.dumps(payload, indent=2)


def render_probe_text(result: ProbeResult) -> str:
    lines = [
        "INFERTOP ACTIVE PROBE",
        f"Endpoint: {result.endpoint}",
        f"Model: {result.model}",
        f"Request: {result.request_id or 'unavailable'}",
        (
            f"Tokens: prompt={result.prompt_tokens if result.prompt_tokens is not None else '?'} "
            "completion="
            f"{result.completion_tokens if result.completion_tokens is not None else '?'}"
        ),
        f"Dominant phase: {result.dominant_phase or 'unavailable'}",
        "",
        result.verdict,
        "Evidence:",
        *(f"- {item}" for item in result.evidence),
        "Remediation:",
        *(f"- {item}" for item in result.remediations),
    ]
    return "\n".join(lines)


def render_probe_json(result: ProbeResult) -> str:
    return json.dumps({"schema_version": 1, "probe": result.to_dict()}, indent=2)
