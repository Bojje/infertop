"""Human and machine-readable reports."""

from __future__ import annotations

import json
from dataclasses import asdict

from infertop.probe import ProbeResult
from infertop.rules import Finding
from infertop.schema import InferenceObservation


def render_text(observation: InferenceObservation, findings: tuple[Finding, ...]) -> str:
    interval = observation.interval_seconds
    sample_summary = (
        f"{interval:.1f}s across {observation.sample_count} samples"
        if interval is not None
        else "single snapshot"
    )
    lines = [
        "INFERTOP",
        f"Source: {observation.current.source}",
        f"Observed: {sample_summary}",
        "",
    ]
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
    payload = {
        "schema_version": 1,
        "source": observation.current.source,
        "sample_count": observation.sample_count,
        "interval_seconds": observation.interval_seconds,
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
