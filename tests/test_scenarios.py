from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from infertop.scenarios import ScenarioError, configured_scenario, run_scenario


def test_scenario_applies_bounds_before_sending_requests() -> None:
    with pytest.raises(ScenarioError, match="request_count must be between"):
        configured_scenario("healthy", request_count=129)
    with pytest.raises(ScenarioError, match="concurrency cannot exceed"):
        configured_scenario("healthy", request_count=2, concurrency=3)
    with pytest.raises(ScenarioError, match="max_tokens must be between"):
        configured_scenario("decode-bound", max_tokens=257)
    with pytest.raises(ScenarioError, match="prompt_words must be between"):
        configured_scenario("healthy", prompt_words=11)


def test_scenario_discovers_model_sends_shaped_requests_and_summarizes() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-0.6B"}]})
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer load-secret"
        assert body["max_tokens"] == 12
        assert body["stream"] is False
        assert len(body["messages"][0]["content"].split()) == 16
        assert "exactly 12 times" in body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "usage": {"prompt_tokens": 18, "completion_tokens": 12},
            },
        )

    scenario = configured_scenario(
        "healthy",
        request_count=3,
        concurrency=2,
        prompt_words=16,
        max_tokens=12,
    )
    result = asyncio.run(
        run_scenario(
            "http://localhost:8000",
            scenario,
            api_key="load-secret",
            transport=httpx.MockTransport(handler),
        )
    )

    assert len(requests) == 4
    assert result.model == "Qwen/Qwen3-0.6B"
    assert result.succeeded == 3
    assert result.failed == 0
    assert result.p50_latency_ms is not None
    assert result.prompt_tokens == 54
    assert result.completion_tokens == 36
    prompts = [json.loads(request.content)["messages"][0]["content"] for request in requests[1:]]
    assert len(set(prompts)) == 3


def test_scenario_records_safe_failures_without_leaking_auth() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Bearer load-secret")

    scenario = configured_scenario("healthy", request_count=1, concurrency=1)
    result = asyncio.run(
        run_scenario(
            "http://localhost:8000",
            scenario,
            model="model",
            api_key="load-secret",
            transport=httpx.MockTransport(handler),
        )
    )

    assert result.failed == 1
    assert result.requests[0].error == "HTTP 429"
    assert "load-secret" not in json.dumps(result.to_dict())
