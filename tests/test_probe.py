from __future__ import annotations

import json

import httpx
import pytest

from infertop.probe import ProbeError, api_base_url, probe_endpoint


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
    assert "--enable-per-request-metrics" in result.remediations[0]


def test_probe_enforces_output_token_safety_bound() -> None:
    with pytest.raises(ProbeError, match="between 1 and 256"):
        probe_endpoint("http://localhost:8000", max_tokens=257)
