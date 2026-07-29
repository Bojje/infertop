from __future__ import annotations

from infertop.mcp_server import probe_inference_endpoint_result
from infertop.probe import ProbeResult


def test_mcp_probe_wrapper_returns_structured_result(monkeypatch) -> None:
    result = ProbeResult(
        endpoint="http://localhost:8000/v1",
        model="model",
        request_id="request-1",
        prompt_tokens=4,
        completion_tokens=1,
        metrics=None,
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
