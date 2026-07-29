"""Normalize vLLM Prometheus samples into the canonical schema."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

from infertop.prometheus import Sample, parse_metrics
from infertop.schema import Distribution, InferenceSnapshot

_ALIASES = {
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


def _values(samples: Iterable[Sample], name: str) -> list[float]:
    return [
        sample.value for sample in samples if sample.name == name and math.isfinite(sample.value)
    ]


def _aggregate(samples: tuple[Sample, ...], aliases: tuple[str, ...], mode: str) -> float | None:
    for name in aliases:
        values = _values(samples, name)
        if values:
            return max(values) if mode == "max" else sum(values)
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


def normalize_vllm(
    text: str,
    *,
    source: str,
    captured_at: float | None = None,
) -> InferenceSnapshot:
    """Normalize one vLLM metrics scrape."""

    samples = parse_metrics(text)
    return InferenceSnapshot(
        source=source,
        captured_at=time.time() if captured_at is None else captured_at,
        requests_running=_aggregate(samples, _ALIASES["running"], "sum"),
        requests_waiting=_aggregate(samples, _ALIASES["waiting"], "sum"),
        kv_cache_usage=_aggregate(samples, _ALIASES["kv_cache"], "max"),
        preemptions_total=_aggregate(samples, _ALIASES["preemptions"], "sum"),
        prefix_cache_queries_total=_aggregate(samples, _ALIASES["prefix_queries"], "sum"),
        prefix_cache_hits_total=_aggregate(samples, _ALIASES["prefix_hits"], "sum"),
        prompt_tokens_total=_aggregate(samples, _ALIASES["prompt_total"], "sum"),
        generation_tokens_total=_aggregate(samples, _ALIASES["generation_total"], "sum"),
        end_to_end_latency_seconds=_histogram(samples, _ALIASES["e2e"]),
        queue_latency_seconds=_histogram(samples, _ALIASES["queue"]),
        time_to_first_token_seconds=_histogram(samples, _ALIASES["ttft"]),
        time_per_output_token_seconds=_histogram(samples, _ALIASES["tpot"]),
        prompt_tokens=_histogram(samples, _ALIASES["prompt"]),
        generation_tokens=_histogram(samples, _ALIASES["generation"]),
        prefill_time_seconds=_histogram(samples, _ALIASES["prefill"]),
        decode_time_seconds=_histogram(samples, _ALIASES["decode"]),
    )
