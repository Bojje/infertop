from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from infertop.collector import CollectionError
from infertop.prometheus_api import (
    MAX_RANGE_SAMPLES,
    build_metric_selector,
    collect_prometheus_range,
    parse_range_time,
    prometheus_query_url,
)
from infertop.report import render_text
from infertop.rules import diagnose

FIXTURE = Path(__file__).parent / "fixtures" / "prometheus" / "kv_thrashing_range.json"


def _transport(payload: dict, received: dict[str, object] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if received is not None:
            received["request"] = request
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def test_saved_prometheus_range_drives_the_same_golden_diagnosis() -> None:
    observation = collect_prometheus_range(
        "http://prometheus:9090",
        start=1000,
        end=1020,
        step_seconds=10,
        labels={"job": "vllm", "instance": "inference:8000"},
        transport=_transport(json.loads(FIXTURE.read_text())),
    )

    assert observation.sample_count == 3
    assert observation.interval_seconds == 20
    assert observation.current.engine == "vllm"
    assert observation.current.source == "prometheus:http://prometheus:9090"
    assert observation.preemptions_delta == 7
    findings = diagnose(observation)
    report = render_text(observation, findings)
    assert findings[0].rule_id == "R3_KV_THRASHING"
    assert "Source: prometheus:http://prometheus:9090" in report
    assert "Blocked:" in report


def test_range_query_is_get_only_filtered_authenticated_and_secret_free() -> None:
    received: dict[str, object] = {}
    collect_prometheus_range(
        "https://metrics.example.com/prometheus",
        start=1000,
        end=1020,
        step_seconds=10,
        labels={"job": 'a"b', "instance": "inference:8000"},
        api_key="secret-token",
        transport=_transport(json.loads(FIXTURE.read_text()), received),
    )

    request = received["request"]
    assert isinstance(request, httpx.Request)
    assert request.method == "GET"
    assert request.url.path == "/prometheus/api/v1/query_range"
    assert request.url.params["start"] == "1000"
    assert request.url.params["end"] == "1020"
    assert request.url.params["step"] == "10"
    assert 'job="a\\"b"' in request.url.params["query"]
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert "secret-token" not in str(request.url)


def test_supported_selector_is_exact_and_rejects_unsafe_label_names() -> None:
    selector = build_metric_selector({"instance": "host\\name\nvalue"})

    assert "vllm:kv_cache_usage_perc" in selector
    assert "sglang_num_retracted_requests_total" in selector
    assert 'instance="host\\\\name\\nvalue"' in selector
    with pytest.raises(CollectionError, match="invalid Prometheus label name"):
        build_metric_selector({'job"=~".*': "bad"})


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    (
        ("http://localhost:9090", "http://localhost:9090/api/v1/query_range"),
        ("http://localhost:9090/api/v1", "http://localhost:9090/api/v1/query_range"),
        (
            "https://example.com/prometheus/api/v1/query_range",
            "https://example.com/prometheus/api/v1/query_range",
        ),
    ),
)
def test_prometheus_query_url_preserves_reverse_proxy_prefix(endpoint: str, expected: str) -> None:
    assert prometheus_query_url(endpoint) == expected


def test_prometheus_range_rejects_multiple_targets() -> None:
    payload = json.loads(FIXTURE.read_text())
    duplicate = json.loads(json.dumps(payload["data"]["result"][0]))
    duplicate["metric"]["instance"] = "other:8000"
    payload["data"]["result"].append(duplicate)

    with pytest.raises(CollectionError, match="matched multiple job/instance targets"):
        collect_prometheus_range(
            "http://prometheus:9090",
            start=1000,
            end=1020,
            step_seconds=10,
            transport=_transport(payload),
        )


def test_prometheus_range_rejects_failed_and_empty_responses() -> None:
    with pytest.raises(CollectionError, match="query failed: execution timed out"):
        collect_prometheus_range(
            "http://prometheus:9090",
            start=1000,
            end=1020,
            step_seconds=10,
            transport=_transport({"status": "error", "error": "execution timed out"}),
        )
    with pytest.raises(CollectionError, match="fewer than two usable snapshots"):
        collect_prometheus_range(
            "http://prometheus:9090",
            start=1000,
            end=1020,
            step_seconds=10,
            transport=_transport(
                {"status": "success", "data": {"resultType": "matrix", "result": []}}
            ),
        )


def test_prometheus_range_is_hard_capped() -> None:
    with pytest.raises(CollectionError, match=f"capped at {MAX_RANGE_SAMPLES}"):
        collect_prometheus_range(
            "http://prometheus:9090",
            start=0,
            end=MAX_RANGE_SAMPLES,
            step_seconds=1,
            transport=_transport({}),
        )


def test_range_time_accepts_unix_and_rfc3339() -> None:
    assert parse_range_time("1000.5") == 1000.5
    assert parse_range_time("1970-01-01T00:16:40Z") == 1000
    with pytest.raises(ValueError, match="UTC offset"):
        parse_range_time("2026-08-01T12:00:00")
