from __future__ import annotations

from infertop.mcp_server import probe_inference_endpoint_result
from infertop.probe import ProbeResult, ProbeTiming


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
