"""Human and machine-readable reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import combinations

from infertop.probe import ProbeResult
from infertop.rules import Finding, assess_diagnostic_coverage
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
    coverage = assess_diagnostic_coverage(observation)
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
        f"Coverage: {coverage.covered_count}/{coverage.total_count} core rules fully covered",
    ]
    if coverage.blocked_rules:
        lines.append(
            "Blocked: "
            + "; ".join(
                f"{gap.rule_id} ({', '.join(gap.reasons)})" for gap in coverage.blocked_rules
            )
        )
    if observation.current.gpus:
        lines.extend(("Hardware: local NVML", *_gpu_summary(observation)))
    if observation.topology is not None:
        indices = observation.tensor_parallel_gpu_indices
        declared = ", ".join(f"GPU{index}" for index in indices)
        lines.append(f"Topology: local nvidia-smi (declared TP GPUs: {declared})")
        for first_gpu, second_gpu in combinations(indices, 2):
            link = observation.topology.link_between(first_gpu, second_gpu)
            kind = link.kind if link is not None else "unknown"
            lines.append(f"GPU {first_gpu} <-> GPU {second_gpu}: {kind}")
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
    coverage = assess_diagnostic_coverage(observation)
    gpus = [
        {
            **asdict(gpu),
            "vram_usage": gpu.vram_usage,
            "power_ratio": gpu.power_ratio,
        }
        for gpu in observation.current.gpus
    ]
    topology = None
    if observation.topology is not None:
        topology = {
            "gpu_indices": observation.topology.gpu_indices,
            "links": [asdict(link) for link in observation.topology.links],
            "tensor_parallel_gpu_indices": observation.tensor_parallel_gpu_indices,
        }
    hardware = None
    if gpus or topology is not None:
        source = "local_nvml"
        if topology is not None:
            source = "local_nvml+nvidia_smi" if gpus else "local_nvidia_smi"
        hardware = {"source": source, "gpus": gpus, "topology": topology}
    payload = {
        "schema_version": 1,
        "engine": observation.current.engine,
        "source": observation.current.source,
        "sample_count": observation.sample_count,
        "interval_seconds": observation.interval_seconds,
        "coverage": {
            "covered_count": coverage.covered_count,
            "total_count": coverage.total_count,
            "covered_rules": coverage.covered_rules,
            "blocked_rules": [asdict(gap) for gap in coverage.blocked_rules],
            "health_verdict_supported": coverage.health_verdict_supported,
        },
        "hardware": hardware,
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
        f"Completion HTTP round trip: {result.timing.client_round_trip_ms:.1f}ms",
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
