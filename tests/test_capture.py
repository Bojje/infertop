from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from infertop.capture import CaptureError, capture_metrics


def test_capture_writes_exact_scrapes_and_provenance_manifest(tmp_path) -> None:
    scrape = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal scrape
        scrape += 1
        assert request.url.path == "/metrics"
        assert request.url.query == b"private=query-secret"
        assert request.headers["Authorization"] == "Bearer metrics-secret"
        return httpx.Response(200, text=f"vllm:num_requests_running {scrape}\n")

    moments = iter(datetime(2026, 8, 1, 10, 0, index, tzinfo=UTC) for index in range(4))
    output = tmp_path / "healthy"
    manifest = capture_metrics(
        "https://example.test?private=query-secret",
        output,
        scenario="healthy",
        sample_count=3,
        interval_seconds=0.25,
        api_key="metrics-secret",
        expected_engine="vllm",
        engine_version="0.10.1",
        model="Qwen/Qwen3-0.6B",
        server_command="vllm serve Qwen/Qwen3-0.6B",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        now=lambda: next(moments),
    )

    assert sleeps == [0.25, 0.25]
    assert (output / "000.prom").read_text() == "vllm:num_requests_running 1\n"
    assert (output / "002.prom").read_text() == "vllm:num_requests_running 3\n"
    payload = json.loads((output / "manifest.json").read_text())
    assert payload["source"] == "https://example.test/metrics"
    assert "query-secret" not in json.dumps(payload)
    assert "metrics-secret" not in json.dumps(payload)
    assert payload["engine"] == "vllm"
    assert payload["engine_version"] == "0.10.1"
    assert payload["model"] == "Qwen/Qwen3-0.6B"
    assert payload["sample_count"] == 3
    assert len(payload["samples"][0]["sha256"]) == 64
    assert manifest.samples[2].file == "002.prom"


def test_capture_refuses_existing_directory(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(CaptureError, match="already exists"):
        capture_metrics(
            "http://localhost:8000",
            output,
            scenario="healthy",
            sample_count=2,
            interval_seconds=1,
        )


def test_capture_rejects_wrong_engine_and_does_not_leak_token(tmp_path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="sglang:num_running_reqs 1\n")

    with pytest.raises(CaptureError, match="expected vllm") as captured:
        capture_metrics(
            "http://localhost:30000?token=query-secret",
            tmp_path / "wrong-engine",
            scenario="healthy",
            sample_count=2,
            interval_seconds=1,
            api_key="metrics-secret",
            expected_engine="vllm",
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
        )

    message = str(captured.value)
    assert "metrics-secret" not in message
    assert "query-secret" not in message
    assert not (tmp_path / "wrong-engine").exists()


def test_capture_redacts_embedded_url_credentials_from_manifest(tmp_path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"vllm:num_requests_running 1\r\n")

    output = tmp_path / "credentials"
    manifest = capture_metrics(
        "https://user:password@example.test",
        output,
        scenario="healthy",
        sample_count=2,
        interval_seconds=1,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert manifest.source == "https://example.test/metrics"
    assert "password" not in (output / "manifest.json").read_text()
    assert (output / "000.prom").read_bytes().endswith(b"\r\n")
