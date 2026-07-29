"""Pure, data-driven diagnostic rules over the canonical schema."""

from __future__ import annotations

from collections.abc import Callable
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


@dataclass(frozen=True)
class Thresholds:
    """Explicit starting thresholds; operators can audit every comparison."""

    high_e2e_p95_seconds: float = 2.0
    high_ttft_p95_seconds: float = 1.0
    high_itl_p95_seconds: float = 0.1
    high_kv_cache_usage: float = 0.90
    critical_kv_cache_usage: float = 0.95
    healthy_prefix_cache_hit_rate: float = 0.25
    long_prompt_p95_tokens: float = 2048
    long_generation_p95_tokens: float = 512
    sequence_skew_ratio: float = 4.0
    underfilled_batch_max_requests: float = 2
    spare_kv_cache_usage: float = 0.50
    batch_ceiling_min_requests: float = 4
    batch_stability_ratio: float = 0.10
    minimum_activity_tokens_per_second: float = 1.0


DEFAULT_THRESHOLDS = Thresholds()
Evaluator = Callable[[InferenceObservation, Thresholds], Finding | None]


@dataclass(frozen=True)
class Rule:
    """Rule metadata plus its pure evaluator."""

    rule_id: str
    inputs: tuple[str, ...]
    evaluate: Evaluator


def _p95(observation: InferenceObservation, name: str) -> float | None:
    distribution = getattr(observation, name)
    return distribution.p95 if distribution is not None else None


def _p50(observation: InferenceObservation, name: str) -> float | None:
    distribution = getattr(observation, name)
    return distribution.p50 if distribution is not None else None


def rule_symptom_isolation(
    observation: InferenceObservation,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Finding | None:
    """R1: use E2E, TTFT, and ITL to locate the slow phase."""

    e2e = _p95(observation, "end_to_end_latency_seconds")
    ttft = _p95(observation, "time_to_first_token_seconds")
    itl = _p95(observation, "time_per_output_token_seconds")
    if e2e is None or ttft is None or itl is None:
        return None
    if e2e < thresholds.high_e2e_p95_seconds or ttft < thresholds.high_ttft_p95_seconds:
        return None
    e2e_p50 = _p50(observation, "end_to_end_latency_seconds")
    ttft_p50 = _p50(observation, "time_to_first_token_seconds")
    itl_p50 = _p50(observation, "time_per_output_token_seconds")
    evidence = (
        (
            f"E2E latency p50/p95: {e2e_p50:.3f}s/{e2e:.3f}s "
            f"(p95 high: >= {thresholds.high_e2e_p95_seconds:.3f}s)"
        ),
        (
            f"TTFT p50/p95: {ttft_p50:.3f}s/{ttft:.3f}s "
            f"(p95 high: >= {thresholds.high_ttft_p95_seconds:.3f}s)"
        ),
        (
            f"ITL p50/p95: {itl_p50:.3f}s/{itl:.3f}s "
            f"(p95 high: >= {thresholds.high_itl_p95_seconds:.3f}s)"
        ),
    )
    if itl < thresholds.high_itl_p95_seconds:
        return Finding(
            rule_id="R1_TTFT_BOUND",
            title="Latency is concentrated before the first token",
            severity=Severity.WARNING,
            score=70,
            summary="High E2E and TTFT with low ITL points to queuing or expensive prefill.",
            evidence=evidence,
            remediations=(
                "Use R2 queue/KV evidence to distinguish overload from compute-bound prefill.",
                "Use R4 prompt-length evidence before changing scheduler configuration.",
            ),
        )
    return Finding(
        rule_id="R1_ALL_PHASES_SLOW",
        title="Prefill and decode are both slow",
        severity=Severity.WARNING,
        score=65,
        summary="High TTFT and ITL indicate pressure across both major inference phases.",
        evidence=evidence,
        remediations=(
            "Inspect prompt lengths and running batch size before tuning.",
            "If spikes coincide with long prompts, tune --max-num-batched-tokens.",
            "Benchmark quantization or faster-memory hardware for persistently high ITL.",
        ),
    )


def rule_saturation(
    observation: InferenceObservation,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Finding | None:
    """R2: distinguish sustained saturation from compute-bound TTFT."""

    current = observation.current
    waiting = current.requests_waiting
    running = current.requests_running
    kv_usage = current.kv_cache_usage
    ttft = _p95(observation, "time_to_first_token_seconds")
    if (
        observation.waiting_is_sustained
        and kv_usage is not None
        and kv_usage >= thresholds.high_kv_cache_usage
    ):
        previous_waiting = observation.previous.requests_waiting if observation.previous else None
        return Finding(
            rule_id="R2_SATURATED",
            title="Server is saturated",
            severity=Severity.CRITICAL,
            score=90,
            summary=(
                "Requests stayed queued throughout the sample window while "
                "KV cache was nearly full."
            ),
            evidence=(
                f"Waiting requests: {previous_waiting:g} -> {waiting:g}",
                f"Running requests: {running:g}" if running is not None else "Running: unavailable",
                (
                    f"KV cache usage: {kv_usage:.1%} "
                    f"(threshold: {thresholds.high_kv_cache_usage:.1%})"
                ),
            ),
            remediations=(
                "Add replicas or reduce admitted request load.",
                "Lower --max-num-seqs if concurrency exceeds sustainable KV capacity.",
                "Use admission control so overload fails predictably instead of queueing.",
            ),
        )
    if (
        waiting == 0
        and ttft is not None
        and ttft >= thresholds.high_ttft_p95_seconds
        and (observation.previous is None or observation.previous.requests_waiting in {None, 0})
    ):
        return Finding(
            rule_id="R2_COMPUTE_BOUND",
            title="High TTFT is not caused by queuing",
            severity=Severity.WARNING,
            score=68,
            summary="TTFT is high while the request queue remains empty.",
            evidence=(
                "Waiting requests: 0",
                f"TTFT p95: {ttft:.3f}s (threshold: {thresholds.high_ttft_p95_seconds:.3f}s)",
            ),
            remediations=(
                "Inspect R4 prompt lengths for prefill-heavy traffic.",
                "Use a smaller task model or reduce retrieved context.",
                "Do not add replicas solely to fix compute time for an individual prompt.",
            ),
        )
    if waiting is not None and waiting > 0 and not observation.waiting_is_sustained:
        return Finding(
            rule_id="R2_QUEUE_OBSERVED",
            title="Requests are queued, but sustained saturation is unproven",
            severity=Severity.INFO,
            score=35,
            summary="One scrape contains waiting work; another positive sample is required.",
            evidence=(f"Current waiting requests: {waiting:g}",),
            remediations=("Collect another scrape before declaring saturation.",),
        )
    return None


def rule_kv_cache_health(
    observation: InferenceObservation,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Finding | None:
    """R3: detect rising preemptions and pinned KV cache."""

    usage = observation.current.kv_cache_usage
    delta = observation.preemptions_delta
    rate = observation.preemptions_per_second
    prefix_hit_rate = observation.prefix_cache_hit_rate
    if delta is not None and delta > 0:
        interval = observation.interval_seconds
        evidence = []
        if usage is not None:
            evidence.append(
                f"KV cache usage: {usage:.1%} "
                f"(pressure threshold: {thresholds.high_kv_cache_usage:.1%})"
            )
        if interval is not None and rate is not None:
            evidence.append(f"Preemptions: +{delta:g} over {interval:.1f}s ({rate:.2f}/s)")
        else:
            evidence.append(f"Preemptions increased by {delta:g}")
        if prefix_hit_rate is not None:
            evidence.append(f"Prefix cache hit rate: {prefix_hit_rate:.1%}")
        return Finding(
            rule_id="R3_KV_THRASHING",
            title="KV cache is thrashing",
            severity=Severity.CRITICAL,
            score=100,
            summary="The preemption counter is rising, proving active memory thrashing.",
            evidence=tuple(evidence),
            remediations=(
                "Lower --max-num-seqs until preemptions stop increasing.",
                "Increase cache capacity with --kv-cache-dtype fp8 where hardware supports it.",
                "Use FP8/AWQ/GPTQ weights or a smaller model to leave more VRAM for KV cache.",
            ),
        )
    if usage is not None and usage >= thresholds.critical_kv_cache_usage:
        evidence = [
            (f"KV cache usage: {usage:.1%} (critical: >= {thresholds.critical_kv_cache_usage:.1%})")
        ]
        remediations = [
            "Lower --max-num-seqs or reduce maximum sequence length.",
            "Use --kv-cache-dtype fp8 where supported to increase cache capacity.",
        ]
        if prefix_hit_rate is not None:
            evidence.append(
                f"Prefix cache hit rate: {prefix_hit_rate:.1%} "
                f"(healthy reuse: >= {thresholds.healthy_prefix_cache_hit_rate:.1%})"
            )
            if prefix_hit_rate < thresholds.healthy_prefix_cache_hit_rate:
                remediations.append(
                    "Put shared system/static context first to improve prefix reuse."
                )
        return Finding(
            rule_id="R3_KV_PRESSURE",
            title="KV cache has almost no headroom",
            severity=Severity.WARNING,
            score=75,
            summary=(
                "KV occupancy is pinned near full, although rising preemptions were not observed."
            ),
            evidence=tuple(evidence),
            remediations=tuple(remediations),
        )
    return None


def rule_sequence_lengths(
    observation: InferenceObservation,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Finding | None:
    """R4: classify clearly asymmetric prompt/output token shapes."""

    prompt = observation.prompt_tokens
    generation = observation.generation_tokens
    prompt_p50 = prompt.p50 if prompt is not None else None
    prompt_p95 = prompt.p95 if prompt is not None else None
    generation_p50 = generation.p50 if generation is not None else None
    generation_p95 = generation.p95 if generation is not None else None
    if prompt_p50 is None or prompt_p95 is None or generation_p50 is None or generation_p95 is None:
        return None
    prefix_hit_rate = observation.prefix_cache_hit_rate
    if (
        prompt_p95 >= thresholds.long_prompt_p95_tokens
        and prompt_p95 >= generation_p95 * thresholds.sequence_skew_ratio
    ):
        evidence = [
            f"Prompt tokens p50/p95: {prompt_p50:.0f}/{prompt_p95:.0f}",
            f"Generation tokens p50/p95: {generation_p50:.0f}/{generation_p95:.0f}",
            f"Prompt/output ratio: {prompt_p95 / max(generation_p95, 1):.1f}x",
        ]
        remediations = [
            "Trim retrieved context and deduplicate RAG passages.",
            "Put shared system/static context first so prefix caching can reuse it.",
            "Consider a smaller task-specific model for prefill-heavy work.",
        ]
        if prefix_hit_rate is not None:
            evidence.append(f"Prefix cache hit rate: {prefix_hit_rate:.1%}")
        return Finding(
            rule_id="R4_PREFILL_BOUND",
            title="Workload is prefill-bound",
            severity=Severity.INFO,
            score=45,
            summary="Long inputs dominate the request shape and put pressure on prefill.",
            evidence=tuple(evidence),
            remediations=tuple(remediations),
        )
    if (
        generation_p95 >= thresholds.long_generation_p95_tokens
        and generation_p95 >= prompt_p95 * thresholds.sequence_skew_ratio
    ):
        return Finding(
            rule_id="R4_DECODE_BOUND",
            title="Workload is decode-bound",
            severity=Severity.INFO,
            score=45,
            summary="Long outputs dominate the request shape and occupy decode slots.",
            evidence=(
                f"Prompt tokens p50/p95: {prompt_p50:.0f}/{prompt_p95:.0f}",
                f"Generation tokens p50/p95: {generation_p50:.0f}/{generation_p95:.0f}",
                f"Output/prompt ratio: {generation_p95 / max(prompt_p95, 1):.1f}x",
            ),
            remediations=(
                "Lower output limits where product behavior permits.",
                "Use stop sequences to avoid unnecessary generation.",
                "Evaluate speculative decoding with --speculative-config.",
            ),
        )
    return None


def rule_batch_efficiency(
    observation: InferenceObservation,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Finding | None:
    """R5: identify conservative batch headroom or a likely concurrency ceiling."""

    if observation.sample_count < 3:
        return None
    running = observation.gauge_values("requests_running")
    waiting = observation.gauge_values("requests_waiting")
    kv_usage = observation.gauge_values("kv_cache_usage")
    token_rate = observation.total_tokens_per_second
    if (
        len(running) != observation.sample_count
        or len(waiting) != observation.sample_count
        or len(kv_usage) != observation.sample_count
        or token_rate is None
        or token_rate < thresholds.minimum_activity_tokens_per_second
    ):
        return None
    prompt_rate = observation.prompt_tokens_per_second or 0.0
    generation_rate = observation.generation_tokens_per_second or 0.0
    running_average = sum(running) / len(running)
    running_spread = max(running) - min(running)
    stable_tolerance = max(1.0, running_average * thresholds.batch_stability_ratio)
    if (
        all(value > 0 for value in waiting)
        and min(running) >= thresholds.batch_ceiling_min_requests
        and running_spread <= stable_tolerance
    ):
        return Finding(
            rule_id="R5_CONCURRENCY_CEILING",
            title="Scheduler appears pinned at a concurrency ceiling",
            severity=Severity.WARNING,
            score=58,
            summary="The running batch stayed flat while waiting work persisted.",
            evidence=(
                f"Running requests across {len(running)} samples: "
                + " -> ".join(f"{value:g}" for value in running),
                "Waiting requests: " + " -> ".join(f"{value:g}" for value in waiting),
                f"Prompt/generation throughput: {prompt_rate:.1f}/{generation_rate:.1f} tokens/s",
                f"KV cache range: {min(kv_usage):.1%}-{max(kv_usage):.1%}",
            ),
            remediations=(
                "Compare the flat running batch with configured --max-num-seqs.",
                "Raise --max-num-seqs only if KV headroom and latency SLOs permit it.",
                "Add replicas when the waiting queue persists at the safe concurrency limit.",
            ),
        )
    if (
        max(running) <= thresholds.underfilled_batch_max_requests
        and all(value == 0 for value in waiting)
        and max(kv_usage) < thresholds.spare_kv_cache_usage
    ):
        return Finding(
            rule_id="R5_BATCH_HEADROOM",
            title="Batch has substantial unused headroom",
            severity=Severity.INFO,
            score=20,
            summary="Active traffic is using tiny batches while KV cache remains mostly free.",
            evidence=(
                f"Running requests across {len(running)} samples: "
                + " -> ".join(f"{value:g}" for value in running),
                "Waiting requests: 0 throughout the observation",
                (
                    f"KV cache max: {max(kv_usage):.1%} "
                    f"(headroom threshold: < {thresholds.spare_kv_cache_usage:.1%})"
                ),
                f"Prompt/generation throughput: {prompt_rate:.1f}/{generation_rate:.1f} tokens/s",
            ),
            remediations=(
                "If throughput matters, increase client concurrency gradually.",
                "Do nothing if this is latency-sensitive traffic with intentionally low demand.",
                "Re-run diagnose under representative peak load before changing server flags.",
            ),
        )
    return None


# Metadata makes required inputs inspectable while evaluators remain ordinary pure functions.
RULES = (
    Rule(
        "R1",
        (
            "end_to_end_latency_seconds",
            "time_to_first_token_seconds",
            "time_per_output_token_seconds",
        ),
        rule_symptom_isolation,
    ),
    Rule(
        "R2",
        ("requests_running", "requests_waiting", "kv_cache_usage"),
        rule_saturation,
    ),
    Rule(
        "R3",
        (
            "kv_cache_usage",
            "preemptions_total",
            "prefix_cache_queries_total",
            "prefix_cache_hits_total",
        ),
        rule_kv_cache_health,
    ),
    Rule("R4", ("prompt_tokens", "generation_tokens"), rule_sequence_lengths),
    Rule(
        "R5",
        (
            "requests_running",
            "requests_waiting",
            "kv_cache_usage",
            "prompt_tokens_total",
            "generation_tokens_total",
        ),
        rule_batch_efficiency,
    ),
)


def diagnose(
    observation: InferenceObservation,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[Finding, ...]:
    """Run and rank all rules deterministically."""

    findings = [
        finding for rule in RULES if (finding := rule.evaluate(observation, thresholds)) is not None
    ]
    if not findings:
        prefix_hit_rate = observation.prefix_cache_hit_rate
        evidence = ["No R1-R5 threshold was crossed."]
        if prefix_hit_rate is not None:
            evidence.append(f"Prefix cache hit rate: {prefix_hit_rate:.1%}")
        findings.append(
            Finding(
                rule_id="HEALTHY",
                title="No diagnosed bottleneck",
                severity=Severity.HEALTHY,
                score=0,
                summary="The available metrics do not indicate a diagnosed bottleneck.",
                evidence=tuple(evidence),
                remediations=("Keep monitoring under representative production traffic.",),
            )
        )
    return tuple(sorted(findings, key=lambda finding: (-finding.score, finding.rule_id)))
