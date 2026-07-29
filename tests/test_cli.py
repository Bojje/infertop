from __future__ import annotations

import json
from pathlib import Path

from infertop.cli import main
from infertop.collector import collect_files
from infertop.probe import ProbeResult, RequestMetrics

FIXTURES = Path(__file__).parent / "fixtures"


def test_diagnose_fixture_prints_ranked_report(capsys) -> None:
    exit_code = main(
        [
            "diagnose",
            str(FIXTURES / "kv_thrashing.prom"),
            "--previous",
            str(FIXTURES / "kv_thrashing_before.prom"),
            "--interval",
            "10",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Engine: vllm" in output
    assert "1. CRITICAL [R3_KV_THRASHING]" in output
    assert "KV cache usage: 97.0%" in output
    assert "Preemptions: +7 over 10.0s (0.70/s)" in output


def test_json_report_is_machine_readable(capsys) -> None:
    exit_code = main(["diagnose", str(FIXTURES / "healthy.prom"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == 1
    assert payload["engine"] == "vllm"
    assert payload["sample_count"] == 1
    assert payload["findings"][0]["rule_id"] == "HEALTHY"


def test_probe_command_prints_per_request_phase_verdict(capsys, monkeypatch) -> None:
    result = ProbeResult(
        endpoint="http://localhost:8000/v1",
        model="model",
        request_id="request-1",
        prompt_tokens=9,
        completion_tokens=2,
        metrics=RequestMetrics(
            time_to_first_token_ms=80,
            generation_time_ms=20,
            queue_time_ms=10,
            mean_itl_ms=10,
            tokens_per_second=20,
        ),
        dominant_phase="prefill/TTFT",
        verdict="This probe spent most of its measured time reaching the first token.",
        evidence=("TTFT after scheduling: 80.0ms",),
        remediations=("Inspect prompt length.",),
    )
    monkeypatch.setattr("infertop.cli.probe_endpoint", lambda *args, **kwargs: result)

    exit_code = main(["probe", "http://localhost:8000"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "INFERTOP ACTIVE PROBE" in output
    assert "prefill" in output


def test_nvml_is_rejected_for_offline_fixture(capsys) -> None:
    exit_code = main(["diagnose", str(FIXTURES / "healthy.prom"), "--nvml"])

    assert exit_code == 2
    assert "--nvml is only valid with a live endpoint" in capsys.readouterr().err


def test_live_nvml_flag_is_forwarded_to_collector(capsys, monkeypatch) -> None:
    observation = collect_files(FIXTURES / "healthy.prom")
    received: dict[str, object] = {}

    def collect(_target: str, **kwargs: object):
        received.update(kwargs)
        return observation

    monkeypatch.setattr("infertop.cli.collect_endpoint", collect)

    exit_code = main(["diagnose", "http://localhost:8000", "--nvml", "--samples", "2"])

    assert exit_code == 0
    assert received["include_nvml"] is True
    assert received["sample_count"] == 2
    assert "Engine: vllm" in capsys.readouterr().out
