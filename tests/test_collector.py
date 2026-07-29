from __future__ import annotations

import asyncio

import httpx
import pytest

from infertop.collector import CollectionError, collect_endpoint, scrape_endpoint_async


def test_live_collector_builds_three_sample_window_and_counter_rates() -> None:
    scrape = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal scrape
        assert request.method == "GET"
        assert request.url.path == "/metrics"
        scrape += 1
        return httpx.Response(
            200,
            text=f"""
            vllm:num_requests_running {scrape}
            vllm:num_requests_waiting 0
            vllm:kv_cache_usage_perc 0.3
            vllm:prompt_tokens_total {scrape * 100}
            vllm:generation_tokens_total {scrape * 200}
            """,
        )

    observation = collect_endpoint(
        "http://localhost:8000",
        interval_seconds=0.001,
        sample_count=3,
        transport=httpx.MockTransport(handler),
    )

    assert scrape == 3
    assert observation.sample_count == 3
    assert observation.current.requests_running == 3
    assert observation.total_tokens_per_second is not None
    assert observation.total_tokens_per_second > 0


def test_live_collector_requires_at_least_two_samples() -> None:
    with pytest.raises(CollectionError, match="at least two"):
        collect_endpoint("http://localhost:8000", sample_count=1)


def test_async_scraper_supports_cancellable_tui_collection() -> None:
    async def exercise() -> None:
        snapshot = await scrape_endpoint_async(
            "http://localhost:8000",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text="vllm:num_requests_running 4",
                )
            ),
        )
        assert snapshot.requests_running == 4

    asyncio.run(exercise())
