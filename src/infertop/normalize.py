"""Normalize engine-specific Prometheus samples into the canonical schema."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

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


def _values(samples: Iterable[Sample], name: str) -> list[float]:
    return [
        sample.value for sample in samples if sample.name == name and math.isfinite(sample.value)
    ]


def _aggregate(samples: tuple[Sample, ...], aliases: tuple[str, ...], mode: str) -> float | None:
    for name in aliases:
        values = _values(samples, name)
        if values:
            if mode == "max":
                return max(values)
            if mode == "average":
                return sum(values) / len(values)
            return sum(values)
    return None


def _histogram(samples: tuple[Sample, ...], aliases: tuple[str, ...]) -> Distribution | None:
    for base_name in aliases:
        bucket_samples = [sample for sample in samples if sample.name == f"{base_name}_bucket"]
        if not bucket_samples:
            continue
        buckets_by_bound: dict[float, float] = {}
        for sample in bucket_samples:
            raw_bound = sample.label("le")
            if raw_bound is None:
                continue
            bound = float(raw_bound)
            buckets_by_bound[bound] = buckets_by_bound.get(bound, 0.0) + sample.value
        count_values = _values(samples, f"{base_name}_count")
        sum_values = _values(samples, f"{base_name}_sum")
        count = sum(count_values) if count_values else buckets_by_bound.get(math.inf, 0.0)
        total = sum(sum_values) if sum_values else None
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
    return InferenceSnapshot(
        source=source,
        captured_at=captured_at,
        engine=engine,
        requests_running=_aggregate(samples, aliases["running"], "sum"),
        requests_waiting=_aggregate(samples, aliases["waiting"], "sum"),
        kv_cache_usage=_aggregate(samples, aliases["kv_cache"], "max"),
        preemptions_total=_aggregate(samples, aliases["preemptions"], "sum"),
        prefix_cache_queries_total=_aggregate(samples, aliases.get("prefix_queries", ()), "sum"),
        prefix_cache_hits_total=_aggregate(samples, aliases.get("prefix_hits", ()), "sum"),
        prefix_cache_hit_rate_gauge=_aggregate(
            samples,
            aliases.get("prefix_hit_rate", ()),
            "average",
        ),
        prompt_tokens_total=_aggregate(samples, aliases["prompt_total"], "sum"),
        generation_tokens_total=_aggregate(samples, aliases["generation_total"], "sum"),
        end_to_end_latency_seconds=_histogram(samples, aliases["e2e"]),
        queue_latency_seconds=_histogram(samples, aliases["queue"]),
        time_to_first_token_seconds=_histogram(samples, aliases["ttft"]),
        time_per_output_token_seconds=_histogram(samples, aliases["tpot"]),
        prompt_tokens=_histogram(samples, aliases["prompt"]),
        generation_tokens=_histogram(samples, aliases["generation"]),
        prefill_time_seconds=_histogram(samples, aliases["prefill"]),
        decode_time_seconds=_histogram(samples, aliases["decode"]),
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
