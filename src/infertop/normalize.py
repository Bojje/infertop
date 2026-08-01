"""Normalize engine-specific Prometheus samples into the canonical schema."""

from __future__ import annotations

import math
import time

from infertop.prometheus import Sample, parse_metrics
from infertop.schema import Distribution, InferenceSnapshot

_VLLM_ALIASES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "preemptions": ("vllm:num_preemptions_total",),
    "prefix_queries": ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
    "prefix_hits": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
    "prompt_total": ("vllm:prompt_tokens_total",),
    "generation_total": ("vllm:generation_tokens_total",),
    "e2e": ("vllm:e2e_request_latency_seconds",),
    "queue": ("vllm:request_queue_time_seconds",),
    "ttft": ("vllm:time_to_first_token_seconds",),
    "tpot": (
        "vllm:inter_token_latency_seconds",
        "vllm:request_time_per_output_token_seconds",
        "vllm:time_per_output_token_seconds",
    ),
    "prompt": ("vllm:request_prompt_tokens", "vllm:prompt_tokens"),
    "generation": ("vllm:request_generation_tokens", "vllm:generation_tokens"),
    "prefill": ("vllm:request_prefill_time_seconds",),
    "decode": ("vllm:request_decode_time_seconds",),
}

_SGLANG_ALIASES = {
    # SGLang has emitted both colon and underscore namespaces. Keep both:
    # https://github.com/sgl-project/sglang/issues/12618
    "running": ("sglang:num_running_reqs", "sglang_num_running_reqs"),
    "waiting": ("sglang:num_queue_reqs", "sglang_num_queue_reqs"),
    "kv_cache": ("sglang:token_usage", "sglang_token_usage"),
    "preemptions": (
        "sglang:num_retracted_requests_total",
        "sglang_num_retracted_requests_total",
    ),
    "prefix_hit_rate": ("sglang:cache_hit_rate", "sglang_cache_hit_rate"),
    "prompt_total": ("sglang:prompt_tokens_total", "sglang_prompt_tokens_total"),
    "generation_total": (
        "sglang:generation_tokens_total",
        "sglang_generation_tokens_total",
    ),
    "e2e": ("sglang:e2e_request_latency_seconds", "sglang_e2e_request_latency_seconds"),
    "queue": ("sglang:queue_time_seconds", "sglang_queue_time_seconds"),
    "ttft": ("sglang:time_to_first_token_seconds", "sglang_time_to_first_token_seconds"),
    "tpot": (
        "sglang:inter_token_latency_seconds",
        "sglang_inter_token_latency_seconds",
        "sglang:time_per_output_token_seconds",
        "sglang_time_per_output_token_seconds",
    ),
    "prompt": ("sglang:prompt_tokens_histogram", "sglang_prompt_tokens_histogram"),
    "generation": (
        "sglang:generation_tokens_histogram",
        "sglang_generation_tokens_histogram",
    ),
    "prefill": (),
    "decode": (),
}


class NormalizationError(ValueError):
    """Raised when metrics cannot be assigned to a supported engine."""


_REPLICATED_SCHEDULER_RANK_LABELS = {"tp_rank", "pp_rank", "moe_ep_rank", "rank"}


def _without_priority_breakdowns(samples: list[Sample]) -> list[Sample]:
    """Prefer SGLang's explicit priority="" total over its per-priority breakdowns."""

    if any(sample.label("priority") == "" for sample in samples):
        return [sample for sample in samples if sample.label("priority") in {None, ""}]
    return samples


def _rank_group_key(sample: Sample) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, value)
        for name, value in sample.labels
        if name not in _REPLICATED_SCHEDULER_RANK_LABELS
    )


def _rank_deduplicated_values(samples: list[Sample]) -> tuple[float, ...]:
    """Use the busiest replica per scheduler group while preserving independent DP shards."""

    groups: dict[tuple[tuple[str, str], ...], list[float]] = {}
    for sample in _without_priority_breakdowns(samples):
        groups.setdefault(_rank_group_key(sample), []).append(sample.value)
    return tuple(max(values) for values in groups.values())


def _aggregate(
    samples: tuple[Sample, ...],
    aliases: tuple[str, ...],
    mode: str,
    *,
    rank_aware: bool = False,
) -> float | None:
    for name in aliases:
        matching = [
            sample for sample in samples if sample.name == name and math.isfinite(sample.value)
        ]
        values = (
            list(_rank_deduplicated_values(matching))
            if rank_aware
            else [sample.value for sample in matching]
        )
        if values:
            if mode == "max":
                return max(values)
            if mode == "average":
                return sum(values) / len(values)
            return sum(values)
    return None


def _sample_sum(samples: list[Sample], *, rank_aware: bool) -> float:
    values = (
        _rank_deduplicated_values(samples)
        if rank_aware
        else tuple(sample.value for sample in samples)
    )
    return sum(values)


def _histogram(
    samples: tuple[Sample, ...],
    aliases: tuple[str, ...],
    *,
    rank_aware: bool = False,
) -> Distribution | None:
    for base_name in aliases:
        bucket_samples = [sample for sample in samples if sample.name == f"{base_name}_bucket"]
        if not bucket_samples:
            continue
        bucket_groups: dict[float, list[Sample]] = {}
        for sample in bucket_samples:
            raw_bound = sample.label("le")
            if raw_bound is None:
                continue
            bound = float(raw_bound)
            bucket_groups.setdefault(bound, []).append(sample)
        buckets_by_bound = {
            bound: _sample_sum(group, rank_aware=rank_aware)
            for bound, group in bucket_groups.items()
        }
        count_samples = [sample for sample in samples if sample.name == f"{base_name}_count"]
        sum_samples = [sample for sample in samples if sample.name == f"{base_name}_sum"]
        count = (
            _sample_sum(count_samples, rank_aware=rank_aware)
            if count_samples
            else buckets_by_bound.get(math.inf, 0.0)
        )
        total = _sample_sum(sum_samples, rank_aware=rank_aware) if sum_samples else None
        return Distribution.from_buckets(
            tuple(buckets_by_bound.items()),
            count=count,
            total=total,
        )
    return None


def _normalize(
    samples: tuple[Sample, ...],
    *,
    aliases: dict[str, tuple[str, ...]],
    engine: str,
    source: str,
    captured_at: float,
) -> InferenceSnapshot:
    rank_aware = engine == "sglang"
    return InferenceSnapshot(
        source=source,
        captured_at=captured_at,
        engine=engine,
        requests_running=_aggregate(samples, aliases["running"], "sum", rank_aware=rank_aware),
        requests_waiting=_aggregate(samples, aliases["waiting"], "sum", rank_aware=rank_aware),
        kv_cache_usage=_aggregate(samples, aliases["kv_cache"], "max", rank_aware=rank_aware),
        preemptions_total=_aggregate(samples, aliases["preemptions"], "sum", rank_aware=rank_aware),
        prefix_cache_queries_total=_aggregate(
            samples, aliases.get("prefix_queries", ()), "sum", rank_aware=rank_aware
        ),
        prefix_cache_hits_total=_aggregate(
            samples, aliases.get("prefix_hits", ()), "sum", rank_aware=rank_aware
        ),
        prefix_cache_hit_rate_gauge=_aggregate(
            samples,
            aliases.get("prefix_hit_rate", ()),
            "average",
            rank_aware=rank_aware,
        ),
        prompt_tokens_total=_aggregate(
            samples, aliases["prompt_total"], "sum", rank_aware=rank_aware
        ),
        generation_tokens_total=_aggregate(
            samples, aliases["generation_total"], "sum", rank_aware=rank_aware
        ),
        end_to_end_latency_seconds=_histogram(samples, aliases["e2e"], rank_aware=rank_aware),
        queue_latency_seconds=_histogram(samples, aliases["queue"], rank_aware=rank_aware),
        time_to_first_token_seconds=_histogram(samples, aliases["ttft"], rank_aware=rank_aware),
        time_per_output_token_seconds=_histogram(samples, aliases["tpot"], rank_aware=rank_aware),
        prompt_tokens=_histogram(samples, aliases["prompt"], rank_aware=rank_aware),
        generation_tokens=_histogram(samples, aliases["generation"], rank_aware=rank_aware),
        prefill_time_seconds=_histogram(samples, aliases["prefill"], rank_aware=rank_aware),
        decode_time_seconds=_histogram(samples, aliases["decode"], rank_aware=rank_aware),
    )


def _captured_at(value: float | None) -> float:
    return time.time() if value is None else value


def normalize_vllm(
    text: str,
    *,
    source: str,
    captured_at: float | None = None,
) -> InferenceSnapshot:
    """Normalize one vLLM metrics scrape."""

    return _normalize(
        parse_metrics(text),
        aliases=_VLLM_ALIASES,
        engine="vllm",
        source=source,
        captured_at=_captured_at(captured_at),
    )


def normalize_sglang(
    text: str,
    *,
    source: str,
    captured_at: float | None = None,
) -> InferenceSnapshot:
    """Normalize one SGLang metrics scrape."""

    return _normalize(
        parse_metrics(text),
        aliases=_SGLANG_ALIASES,
        engine="sglang",
        source=source,
        captured_at=_captured_at(captured_at),
    )


def normalize_metrics(
    text: str,
    *,
    source: str,
    captured_at: float | None = None,
) -> InferenceSnapshot:
    """Detect a supported engine and normalize one metrics scrape."""

    samples = parse_metrics(text)
    names = {sample.name for sample in samples}
    engines = {
        engine
        for engine, prefixes in (
            ("vllm", ("vllm:", "vllm_")),
            ("sglang", ("sglang:", "sglang_")),
        )
        if any(name.startswith(prefixes) for name in names)
    }
    if len(engines) != 1:
        detected = ", ".join(sorted(engines)) or "none"
        raise NormalizationError(
            f"expected metrics from exactly one supported engine; detected: {detected}"
        )
    engine = engines.pop()
    aliases = _VLLM_ALIASES if engine == "vllm" else _SGLANG_ALIASES
    return _normalize(
        samples,
        aliases=aliases,
        engine=engine,
        source=source,
        captured_at=_captured_at(captured_at),
    )
