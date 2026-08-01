from __future__ import annotations

import json

import httpx
import pytest

from infertop.probe import (
    MAX_PROBE_PROMPT_CHARACTERS,
    ProbeError,
    RequestMetrics,
    api_base_url,
    correlate_probe_timing,
    probe_endpoint,
    probe_endpoint_repeated,
)
from infertop.report import render_probe_json, render_probe_text


def test_api_base_url_accepts_server_metrics_and_v1_urls() -> None:
    assert api_base_url("http://localhost:8000") == "http://localhost:8000/v1"
    assert api_base_url("http://localhost:8000/metrics") == "http://localhost:8000/v1"
    assert api_base_url("https://example.test/prefix/v1") == "https://example.test/prefix/v1"
    assert api_base_url("https://example.test?token=secret") == "https://example.test/v1"


def test_probe_discovers_model_sends_bounded_request_and_parses_metrics() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-0.6B"}]})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "Qwen/Qwen3-0.6B"
        assert body["max_tokens"] == 8
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
                "metrics": {
                    "time_to_first_token_ms": 85.2,
                    "generation_time_ms": 20.0,
                    "queue_time_ms": 12.3,
                    "mean_itl_ms": 9.1,
                    "tokens_per_second": 19.0,
                },
            },
        )

    result = probe_endpoint(
        "http://localhost:8000",
        transport=httpx.MockTransport(handler),
    )

    assert len(requests) == 2
    assert result.model == "Qwen/Qwen3-0.6B"
    assert result.dominant_phase == "prefill/TTFT"
    assert result.metrics is not None
    assert result.metrics.queue_time_ms == 12.3
    assert "Output throughput: 19.0 tokens/s" in result.evidence


def test_probe_explains_missing_per_request_metrics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "usage": {"prompt_tokens": 8, "completion_tokens": 1},
            },
        )

    result = probe_endpoint(
        "http://localhost:8000",
        model="model",
        transport=httpx.MockTransport(handler),
    )

    assert result.metrics is None
    assert result.timing.server_accounted_ms is None
    assert "--enable-per-request-metrics" in result.remediations[0]


def test_probe_enforces_output_token_safety_bound() -> None:
    with pytest.raises(ProbeError, match="between 1 and 256"):
        probe_endpoint("http://localhost:8000", max_tokens=257)


def test_probe_enforces_prompt_safety_bound() -> None:
    with pytest.raises(ProbeError, match="prompt must contain"):
        probe_endpoint("http://localhost:8000", prompt="x" * (MAX_PROBE_PROMPT_CHARACTERS + 1))


def test_repeated_probe_summarizes_nearest_rank_percentiles(monkeypatch) -> None:
    moments = iter((0.0, 0.1, 1.0, 1.2, 2.0, 2.5))
    monkeypatch.setattr("infertop.probe._monotonic_seconds", lambda: next(moments))
    completions = 0
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal completions
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        completions += 1
        return httpx.Response(
            200,
            json={
                "id": f"request-{completions}",
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                "metrics": {
                    "queue_time_ms": completions * 10,
                    "time_to_first_token_ms": completions * 20,
                    "generation_time_ms": completions * 30,
                    "mean_itl_ms": completions * 2,
                    "tokens_per_second": 100 / completions,
                },
            },
        )

    result = probe_endpoint_repeated(
        "http://localhost:8000",
        request_count=3,
        max_tokens=8,
        transport=httpx.MockTransport(handler),
    )

    assert len(requests) == 4
    assert result.request_count == 3
    assert result.requested_output_token_ceiling == 24
    assert result.reported_prompt_tokens == 12
    assert result.reported_completion_tokens == 6
    assert result.client_round_trip_ms.p50 == pytest.approx(200)
    assert result.client_round_trip_ms.p95 == pytest.approx(500)
    assert result.metric_percentiles("queue_time_ms").p50 == 20
    assert result.metric_percentiles("queue_time_ms").p95 == 30
    assert result.metrics_sample_count == 3
    assert result.dominant_phase_counts == {"decode": 1, "outside engine": 2}
    assert "Output ceiling: 8/request; 24 total" in render_probe_text(result)
    assert "probe_run" in json.loads(render_probe_json(result))


def test_repeated_probe_enforces_request_and_total_token_caps_before_traffic() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent")

    transport = httpx.MockTransport(handler)
    with pytest.raises(ProbeError, match="request_count must be between"):
        probe_endpoint_repeated(
            "http://localhost:8000",
            request_count=11,
            transport=transport,
        )
    with pytest.raises(ProbeError, match="output ceiling exceeds 1024"):
        probe_endpoint_repeated(
            "http://localhost:8000",
            request_count=5,
            max_tokens=256,
            transport=transport,
        )


def test_correlates_client_round_trip_with_server_accounted_time() -> None:
    timing = correlate_probe_timing(
        RequestMetrics(
            queue_time_ms=50,
            time_to_first_token_ms=100,
            generation_time_ms=200,
        ),
        client_round_trip_ms=1000,
    )

    assert timing.server_accounted_ms == pytest.approx(350)
    assert timing.outside_engine_ms == pytest.approx(650)
    assert timing.outside_engine_ratio == pytest.approx(0.65)


def test_probe_diagnoses_significant_time_outside_engine_phases(monkeypatch) -> None:
    moments = iter((10.0, 11.0))
    monkeypatch.setattr("infertop.probe._monotonic_seconds", lambda: next(moments))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "usage": {"prompt_tokens": 9, "completion_tokens": 20},
                "metrics": {
                    "time_to_first_token_ms": 100,
                    "generation_time_ms": 200,
                    "queue_time_ms": 50,
                    "mean_itl_ms": 10,
                    "tokens_per_second": 66,
                },
            },
        )

    result = probe_endpoint(
        "http://localhost:8000",
        model="model",
        transport=httpx.MockTransport(handler),
    )

    assert result.dominant_phase == "outside engine"
    assert result.timing.outside_engine_ms == pytest.approx(650)
    assert "unattributed time" in result.remediations[-1]


def test_significant_residual_does_not_override_larger_engine_phase(monkeypatch) -> None:
    moments = iter((20.0, 20.45))
    monkeypatch.setattr("infertop.probe._monotonic_seconds", lambda: next(moments))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "metrics": {
                    "time_to_first_token_ms": 40,
                    "generation_time_ms": 300,
                    "queue_time_ms": 10,
                }
            },
        )

    result = probe_endpoint(
        "http://localhost:8000",
        model="model",
        transport=httpx.MockTransport(handler),
    )

    assert result.timing.outside_engine_ms == pytest.approx(100)
    assert result.dominant_phase == "decode"
