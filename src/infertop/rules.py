"""Pure diagnostic rules over the canonical schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from infertop.schema import InferenceObservation


class Severity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    HEALTHY = "healthy"


@dataclass(frozen=True)
class Finding:
    rule_id: str
    title: str
    severity: Severity
    score: int
    summary: str
    evidence: tuple[str, ...]
    remediations: tuple[str, ...]


def rule_queue_saturation(observation: InferenceObservation) -> Finding | None:
    """R1: detect work accumulating behind the scheduler."""

    current = observation.current
    waiting = current.requests_waiting
    if waiting is None or waiting < 1:
        return None
    running = current.requests_running
    queue_p95 = (
        current.queue_latency_seconds.p95 if current.queue_latency_seconds is not None else None
    )
    saturated = (running is not None and waiting >= max(running, 1)) or (
        queue_p95 is not None and queue_p95 >= 0.25
    )
    evidence = [f"Waiting requests: {waiting:g}"]
    if running is not None:
        evidence.append(f"Running requests: {running:g}")
    if queue_p95 is not None:
        evidence.append(f"Queue latency p95: {queue_p95:.3f}s")
    return Finding(
        rule_id="R1_QUEUE_SATURATED",
        title="Request queue is saturated" if saturated else "Requests are beginning to queue",
        severity=Severity.CRITICAL if saturated else Severity.WARNING,
        score=90 if saturated else 60,
        summary=(
            "Demand is accumulating faster than the scheduler can admit work."
            if saturated
            else "Some requests are waiting, but the queue is not yet deeply saturated."
        ),
        evidence=tuple(evidence),
        remediations=(
            "Reduce request concurrency or add a replica.",
            "Cap oversized prompts/outputs and retry with admission control.",
            "Check KV pressure before increasing scheduler concurrency.",
        ),
    )


def rule_kv_thrashing(observation: InferenceObservation) -> Finding | None:
    """R2: require both high KV pressure and observed preemption growth."""

    usage = observation.current.kv_cache_usage
    delta = observation.preemptions_delta
    rate = observation.preemptions_per_second
    if usage is None or usage < 0.90 or delta is None or delta <= 0:
        return None
    interval = observation.interval_seconds
    evidence = [f"KV cache usage: {usage:.1%} (threshold: 90.0%)"]
    if interval is not None and rate is not None:
        evidence.append(f"Preemptions: +{delta:g} over {interval:.1f}s ({rate:.2f}/s)")
    else:
        evidence.append(f"Preemptions increased by {delta:g}")
    return Finding(
        rule_id="R2_KV_THRASHING",
        title="KV cache is thrashing",
        severity=Severity.CRITICAL,
        score=100,
        summary="KV cache is nearly full while requests are being preempted.",
        evidence=tuple(evidence),
        remediations=(
            "Reduce max model length or concurrent sequences.",
            "Increase KV-cache headroom or use a smaller/quantized model.",
            "Confirm improvement by checking that preemptions stop increasing.",
        ),
    )


def rule_workload_shape(observation: InferenceObservation) -> Finding | None:
    """R3: classify clearly asymmetric prompt/output token shapes."""

    prompt = observation.current.prompt_tokens
    generation = observation.current.generation_tokens
    prompt_p90 = prompt.p90 if prompt is not None else None
    generation_p90 = generation.p90 if generation is not None else None
    if prompt_p90 is None or generation_p90 is None:
        return None
    if prompt_p90 >= 1024 and prompt_p90 >= generation_p90 * 4:
        return Finding(
            rule_id="R3_PREFILL_BOUND",
            title="Workload is prefill-bound",
            severity=Severity.INFO,
            score=45,
            summary="Long inputs dominate the request shape and put pressure on prefill.",
            evidence=(
                f"Prompt tokens p90: {prompt_p90:.0f}",
                f"Generation tokens p90: {generation_p90:.0f}",
                f"Prompt/output ratio: {prompt_p90 / max(generation_p90, 1):.1f}x",
            ),
            remediations=(
                "Trim retrieved context and deduplicate RAG passages.",
                "Enable or improve prefix caching for repeated prompt prefixes.",
                "Batch prefill carefully and track time-to-first-token.",
            ),
        )
    if generation_p90 >= 256 and generation_p90 >= prompt_p90 * 4:
        return Finding(
            rule_id="R3_DECODE_BOUND",
            title="Workload is decode-bound",
            severity=Severity.INFO,
            score=45,
            summary="Long outputs dominate the request shape and occupy decode slots.",
            evidence=(
                f"Prompt tokens p90: {prompt_p90:.0f}",
                f"Generation tokens p90: {generation_p90:.0f}",
                f"Output/prompt ratio: {generation_p90 / max(prompt_p90, 1):.1f}x",
            ),
            remediations=(
                "Lower output limits where product behavior permits.",
                "Use stop sequences and discourage unnecessarily verbose generations.",
                "Track time-per-output-token while tuning decode concurrency.",
            ),
        )
    return None


RULES = (rule_queue_saturation, rule_kv_thrashing, rule_workload_shape)


def diagnose(observation: InferenceObservation) -> tuple[Finding, ...]:
    """Run and rank all rules deterministically."""

    findings = [finding for rule in RULES if (finding := rule(observation)) is not None]
    if not findings:
        findings.append(
            Finding(
                rule_id="HEALTHY",
                title="No diagnosed bottleneck",
                severity=Severity.HEALTHY,
                score=0,
                summary="The available metrics do not trigger R1-R3.",
                evidence=("No queue saturation, measured KV thrashing, or extreme token skew.",),
                remediations=("Keep monitoring under representative production traffic.",),
            )
        )
    return tuple(sorted(findings, key=lambda finding: (-finding.score, finding.rule_id)))
