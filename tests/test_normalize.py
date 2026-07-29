from __future__ import annotations

from infertop.normalize import normalize_vllm


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
