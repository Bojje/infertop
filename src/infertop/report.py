"""Human and machine-readable reports."""

from __future__ import annotations

import json
from dataclasses import asdict

from infertop.rules import Finding
from infertop.schema import InferenceObservation


def render_text(observation: InferenceObservation, findings: tuple[Finding, ...]) -> str:
    interval = observation.interval_seconds
    sample_summary = (
        f"{interval:.1f}s across 2 samples" if interval is not None else "single snapshot"
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
        "interval_seconds": observation.interval_seconds,
        "findings": [asdict(finding) for finding in findings],
    }
    return json.dumps(payload, indent=2)
