from __future__ import annotations

from pathlib import Path

from infertop.collector import collect_files
from infertop.mcp_server import (
    diagnose_endpoint_result,
    diagnose_prometheus_range_result,
    probe_inference_endpoint_result,
)
from infertop.probe import ProbeResult, ProbeTiming

FIXTURES = Path(__file__).parent / "fixtures"


def test_mcp_diagnosis_reads_metrics_token_from_named_environment(monkeypatch) -> None:
    observation = collect_files(FIXTURES / "healthy.prom")
    received: dict[str, object] = {}

    def collect(_endpoint: str, **kwargs: object):
        received.update(kwargs)
        return observation

    monkeypatch.setattr("infertop.mcp_server.collect_endpoint", collect)
    monkeypatch.setenv("METRICS_TOKEN", "metrics-secret")

    payload = diagnose_endpoint_result(
        "http://localhost:8000",
        api_key_env="METRICS_TOKEN",
        tensor_parallel_gpu_indices=[0, 1],
    )

    assert received["api_key"] == "metrics-secret"
    assert received["tensor_parallel_gpu_indices"] == (0, 1)
    assert "metrics-secret" not in repr(payload)


def test_mcp_probe_wrapper_returns_structured_result(monkeypatch) -> None:
    result = ProbeResult(
        endpoint="http://localhost:8000/v1",
        model="model",
        request_id="request-1",
        prompt_tokens=4,
        completion_tokens=1,
        metrics=None,
        timing=ProbeTiming(
            client_round_trip_ms=100,
            server_accounted_ms=None,
            outside_engine_ms=None,
            outside_engine_ratio=None,
        ),
        dominant_phase=None,
        verdict="No timing metrics.",
        evidence=("Unavailable",),
        remediations=("Enable metrics.",),
    )
    monkeypatch.setattr(
        "infertop.mcp_server.probe_endpoint",
        lambda *args, **kwargs: result,
    )

    payload = probe_inference_endpoint_result("http://localhost:8000")

    assert payload["model"] == "model"
    assert payload["request_id"] == "request-1"
    assert payload["timing"]["client_round_trip_ms"] == 100


def test_mcp_prometheus_wrapper_reads_token_and_parses_range(monkeypatch) -> None:
    observation = collect_files(FIXTURES / "healthy.prom")
    received: dict[str, object] = {}

    def collect(_endpoint: str, **kwargs: object):
        received.update(kwargs)
        return observation

    monkeypatch.setattr("infertop.mcp_server.collect_prometheus_range", collect)
    monkeypatch.setenv("PROMETHEUS_TOKEN", "prometheus-secret")

    payload = diagnose_prometheus_range_result(
        "http://prometheus:9090",
        start="1970-01-01T00:16:40Z",
        end="1020",
        step_seconds=10,
        labels={"instance": "inference:8000"},
        api_key_env="PROMETHEUS_TOKEN",
    )

    assert received["start"] == 1000
    assert received["end"] == 1020
    assert received["labels"] == {"instance": "inference:8000"}
    assert received["api_key"] == "prometheus-secret"
    assert "prometheus-secret" not in repr(payload)
