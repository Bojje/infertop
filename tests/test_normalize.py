from __future__ import annotations

from pathlib import Path

import pytest

from infertop.normalize import (
    NormalizationError,
    normalize_metrics,
    normalize_sglang,
    normalize_vllm,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_aggregates_workers_and_interpolates_histogram_quantiles() -> None:
    snapshot = normalize_vllm(
        """
        vllm:num_requests_running{worker="0"} 2
        vllm:num_requests_running{worker="1"} 3
        vllm:gpu_cache_usage_perc{worker="0"} 0.4
        vllm:gpu_cache_usage_perc{worker="1"} 0.7
        vllm:prefix_cache_queries 100
        vllm:prefix_cache_hits_total 30
        vllm:request_prompt_tokens_bucket{worker="0",le="10"} 5
        vllm:request_prompt_tokens_bucket{worker="1",le="10"} 5
        vllm:request_prompt_tokens_bucket{worker="0",le="20"} 10
        vllm:request_prompt_tokens_bucket{worker="1",le="20"} 10
        vllm:request_prompt_tokens_bucket{worker="0",le="+Inf"} 10
        vllm:request_prompt_tokens_bucket{worker="1",le="+Inf"} 10
        vllm:request_prompt_tokens_count{worker="0"} 10
        vllm:request_prompt_tokens_count{worker="1"} 10
        """,
        source="fixture",
        captured_at=123,
    )

    assert snapshot.requests_running == 5
    assert snapshot.kv_cache_usage == 0.7
    assert snapshot.prefix_cache_queries_total == 100
    assert snapshot.prefix_cache_hits_total == 30
    assert snapshot.prompt_tokens is not None
    assert snapshot.prompt_tokens.p50 == 10
    assert snapshot.prompt_tokens.p90 == 18
    assert snapshot.engine == "vllm"


def test_normalizes_current_sglang_metrics_into_canonical_schema() -> None:
    snapshot = normalize_sglang(
        """
        sglang:num_running_reqs{model_name="Qwen/Qwen3-0.6B"} 6
        sglang:num_queue_reqs{model_name="Qwen/Qwen3-0.6B"} 2
        sglang:token_usage{model_name="Qwen/Qwen3-0.6B"} 0.72
        sglang:num_retracted_requests_total{model_name="Qwen/Qwen3-0.6B"} 4
        sglang:cache_hit_rate{model_name="Qwen/Qwen3-0.6B"} 0.35
        sglang:prompt_tokens_total{model_name="Qwen/Qwen3-0.6B"} 1000
        sglang:generation_tokens_total{model_name="Qwen/Qwen3-0.6B"} 500
        sglang:prompt_tokens_histogram_bucket{le="100"} 4
        sglang:prompt_tokens_histogram_bucket{le="1000"} 10
        sglang:prompt_tokens_histogram_bucket{le="+Inf"} 10
        sglang:prompt_tokens_histogram_count 10
        """,
        source="fixture",
        captured_at=123,
    )

    assert snapshot.engine == "sglang"
    assert snapshot.requests_running == 6
    assert snapshot.requests_waiting == 2
    assert snapshot.kv_cache_usage == pytest.approx(0.72)
    assert snapshot.preemptions_total == 4
    assert snapshot.prefix_cache_hit_rate_gauge == pytest.approx(0.35)
    assert snapshot.prompt_tokens is not None
    assert snapshot.prompt_tokens.p95 == pytest.approx(925)


def test_auto_detection_accepts_historical_sglang_underscore_namespace() -> None:
    snapshot = normalize_metrics(
        """
        sglang_num_running_reqs 3
        sglang_num_queue_reqs 0
        sglang_token_usage 0.4
        """,
        source="fixture",
        captured_at=123,
    )

    assert snapshot.engine == "sglang"
    assert snapshot.requests_running == 3


def test_sglang_deduplicates_scheduler_ranks_and_sums_dp_shards() -> None:
    snapshot = normalize_sglang(
        (FIXTURES / "sglang_multirank.prom").read_text(),
        source="fixture",
        captured_at=123,
    )

    assert snapshot.requests_running == 7
    assert snapshot.requests_waiting == 3
    assert snapshot.kv_cache_usage == pytest.approx(0.96)
    assert snapshot.preemptions_total == 15
    assert snapshot.prefix_cache_hit_rate_gauge == pytest.approx(0.30)
    assert snapshot.prompt_tokens_total == 1000
    assert snapshot.generation_tokens_total == 550


def test_sglang_ranked_histograms_match_equivalent_single_rank_histogram() -> None:
    ranked = normalize_sglang(
        """
        sglang:prompt_tokens_histogram_bucket{tp_rank="0",dp_rank="0",le="100"} 4
        sglang:prompt_tokens_histogram_bucket{tp_rank="1",dp_rank="0",le="100"} 4
        sglang:prompt_tokens_histogram_bucket{tp_rank="0",dp_rank="0",le="1000"} 10
        sglang:prompt_tokens_histogram_bucket{tp_rank="1",dp_rank="0",le="1000"} 10
        sglang:prompt_tokens_histogram_bucket{tp_rank="0",dp_rank="0",le="+Inf"} 10
        sglang:prompt_tokens_histogram_bucket{tp_rank="1",dp_rank="0",le="+Inf"} 10
        sglang:prompt_tokens_histogram_count{tp_rank="0",dp_rank="0"} 10
        sglang:prompt_tokens_histogram_count{tp_rank="1",dp_rank="0"} 10
        """,
        source="ranked",
    )
    single = normalize_sglang(
        """
        sglang:prompt_tokens_histogram_bucket{le="100"} 4
        sglang:prompt_tokens_histogram_bucket{le="1000"} 10
        sglang:prompt_tokens_histogram_bucket{le="+Inf"} 10
        sglang:prompt_tokens_histogram_count 10
        """,
        source="single",
    )

    assert ranked.prompt_tokens == single.prompt_tokens


def test_historical_sglang_namespace_uses_the_same_rank_deduplication() -> None:
    snapshot = normalize_metrics(
        (FIXTURES / "sglang_multirank_historical.prom").read_text(),
        source="fixture",
    )

    assert snapshot.engine == "sglang"
    assert snapshot.requests_running == 7
    assert snapshot.preemptions_total == 15


def test_auto_detection_rejects_unknown_or_mixed_engine_metrics() -> None:
    with pytest.raises(NormalizationError, match="detected: none"):
        normalize_metrics("process_cpu_seconds_total 1", source="fixture")

    with pytest.raises(NormalizationError, match="sglang, vllm"):
        normalize_metrics(
            "vllm:num_requests_running 1\nsglang:num_running_reqs 1",
            source="fixture",
        )
