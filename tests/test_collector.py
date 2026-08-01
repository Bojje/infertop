from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from infertop.collector import (
    CollectionError,
    collect_endpoint,
    collect_file_series,
    scrape_endpoint_async,
)
from infertop.schema import GpuDeviceSnapshot, GpuTopology, GpuTopologyLink

FIXTURES = Path(__file__).parent / "fixtures"


def test_live_collector_builds_three_sample_window_and_counter_rates(monkeypatch) -> None:
    scrape = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal scrape
        assert request.method == "GET"
        assert request.url.path == "/metrics"
        assert request.headers["Authorization"] == "Bearer metrics-secret"
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

    gpu = GpuDeviceSnapshot(index=0, name="GPU", uuid="GPU-test", gpu_utilization=0.5)
    monkeypatch.setattr("infertop.collector.collect_nvml_gpus", lambda: (gpu,))

    observation = collect_endpoint(
        "http://localhost:8000",
        interval_seconds=0.001,
        sample_count=3,
        include_nvml=True,
        api_key="metrics-secret",
        transport=httpx.MockTransport(handler),
    )

    assert scrape == 3
    assert observation.sample_count == 3
    assert observation.current.requests_running == 3
    assert observation.current.gpus == (gpu,)
    assert observation.gpu_average_values("gpu_utilization") == (0.5, 0.5, 0.5)
    assert observation.total_tokens_per_second is not None
    assert observation.total_tokens_per_second > 0


def test_live_collector_requires_at_least_two_samples() -> None:
    with pytest.raises(CollectionError, match="at least two"):
        collect_endpoint("http://localhost:8000", sample_count=1)


def test_collection_error_does_not_leak_bearer_token() -> None:
    with pytest.raises(CollectionError) as captured:
        collect_endpoint(
            "http://localhost:8000",
            sample_count=2,
            api_key="metrics-secret",
            transport=httpx.MockTransport(lambda _request: httpx.Response(401)),
        )

    assert "metrics-secret" not in str(captured.value)


def test_live_source_and_errors_redact_url_credentials_and_query_values() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.query == b"tenant=query-secret"
        return httpx.Response(200, text="vllm:num_requests_running 1")

    observation = collect_endpoint(
        "https://user:password@example.test?tenant=query-secret",
        interval_seconds=0.001,
        sample_count=2,
        transport=httpx.MockTransport(handler),
    )
    assert observation.current.source == "https://example.test/metrics"

    with pytest.raises(CollectionError) as captured:
        collect_endpoint(
            "https://user:password@example.test?tenant=query-secret",
            sample_count=2,
            transport=httpx.MockTransport(lambda _request: httpx.Response(401)),
        )
    message = str(captured.value)
    assert "password" not in message
    assert "query-secret" not in message


def test_offline_series_interval_is_between_adjacent_snapshots() -> None:
    observation = collect_file_series(
        (
            FIXTURES / "batch_headroom_before.prom",
            FIXTURES / "batch_headroom_middle.prom",
            FIXTURES / "batch_headroom.prom",
        ),
        interval_seconds=10,
    )

    assert observation.sample_count == 3
    assert observation.interval_seconds == 20
    assert tuple(snapshot.captured_at for snapshot in observation.snapshots) == (0, 10, 20)


def test_async_scraper_supports_cancellable_tui_collection() -> None:
    async def exercise() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer metrics-secret"
            return httpx.Response(200, text="vllm:num_requests_running 4")

        snapshot = await scrape_endpoint_async(
            "http://localhost:8000",
            api_key="metrics-secret",
            transport=httpx.MockTransport(handler),
        )
        assert snapshot.requests_running == 4

    asyncio.run(exercise())


def test_live_collector_queries_topology_once_for_explicit_tp_group(monkeypatch) -> None:
    topology = GpuTopology(
        gpu_indices=(0, 1),
        links=(GpuTopologyLink(first_gpu=0, second_gpu=1, kind="NV4"),),
    )
    topology_queries = 0

    def collect_topology() -> GpuTopology:
        nonlocal topology_queries
        topology_queries += 1
        return topology

    monkeypatch.setattr("infertop.collector.collect_nvidia_topology", collect_topology)
    observation = collect_endpoint(
        "http://localhost:8000",
        interval_seconds=0.001,
        sample_count=2,
        tensor_parallel_gpu_indices=(0, 1),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, text="vllm:num_requests_running 1")
        ),
    )

    assert topology_queries == 1
    assert observation.topology == topology
    assert observation.tensor_parallel_gpu_indices == (0, 1)


def test_live_collector_rejects_unknown_tp_gpu_before_metrics_request(monkeypatch) -> None:
    monkeypatch.setattr(
        "infertop.collector.collect_nvidia_topology",
        lambda: GpuTopology(gpu_indices=(0,)),
    )
    metrics_requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal metrics_requests
        metrics_requests += 1
        return httpx.Response(200, text="vllm:num_requests_running 1")

    with pytest.raises(ValueError, match="unknown topology devices: GPU1"):
        collect_endpoint(
            "http://localhost:8000",
            sample_count=2,
            tensor_parallel_gpu_indices=(0, 1),
            transport=httpx.MockTransport(handler),
        )

    assert metrics_requests == 0
