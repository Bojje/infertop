from __future__ import annotations

import json
from pathlib import Path

import pytest

from infertop.cli import diagnosis_exit_code, main
from infertop.collector import collect_files
from infertop.probe import ProbeResult, ProbeTiming, RequestMetrics
from infertop.rules import Finding, Severity

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
    assert "Coverage:" in output
    assert "1. CRITICAL [R3_KV_THRASHING]" in output
    assert "KV cache usage: 97.0%" in output
    assert "Preemptions: +7 over 10.0s (0.70/s)" in output


def _finding(severity: Severity) -> Finding:
    return Finding(
        rule_id="TEST",
        title="Test",
        severity=severity,
        score=1,
        summary="Test",
        evidence=("Test",),
        remediations=("Test",),
    )


@pytest.mark.parametrize(
    ("fail_on", "severity", "expected"),
    (
        (None, Severity.CRITICAL, 0),
        ("critical", Severity.CRITICAL, 1),
        ("critical", Severity.WARNING, 0),
        ("warning", Severity.CRITICAL, 1),
        ("warning", Severity.WARNING, 1),
        ("warning", Severity.INFO, 0),
        ("info", Severity.INFO, 1),
        ("info", Severity.HEALTHY, 0),
    ),
)
def test_diagnosis_exit_code_policy(
    fail_on: str | None,
    severity: Severity,
    expected: int,
) -> None:
    assert diagnosis_exit_code((_finding(severity),), fail_on) == expected


def test_diagnosis_exit_code_accepts_one_pass_findings() -> None:
    findings = (_finding(severity) for severity in (Severity.INFO, Severity.CRITICAL))

    assert diagnosis_exit_code(findings, "critical") == 1


def test_fail_on_critical_returns_one_after_printing_report(capsys) -> None:
    exit_code = main(
        [
            "diagnose",
            str(FIXTURES / "kv_thrashing.prom"),
            "--previous",
            str(FIXTURES / "kv_thrashing_before.prom"),
            "--interval",
            "10",
            "--fail-on",
            "critical",
        ]
    )

    assert exit_code == 1
    assert "R3_KV_THRASHING" in capsys.readouterr().out


def test_replays_three_file_series_for_r5_diagnosis(capsys) -> None:
    exit_code = main(
        [
            "diagnose",
            str(FIXTURES / "batch_headroom.prom"),
            "--previous",
            str(FIXTURES / "batch_headroom_before.prom"),
            "--intermediate",
            str(FIXTURES / "batch_headroom_middle.prom"),
            "--interval",
            "10",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Observed: 20.0s across 3 samples" in output
    assert "1. INFO [R5_BATCH_HEADROOM]" in output


def test_intermediate_fixture_requires_previous(capsys) -> None:
    exit_code = main(
        [
            "diagnose",
            str(FIXTURES / "batch_headroom.prom"),
            "--intermediate",
            str(FIXTURES / "batch_headroom_middle.prom"),
        ]
    )

    assert exit_code == 2
    assert "--intermediate requires --previous" in capsys.readouterr().err


def test_intermediate_fixture_is_rejected_for_live_endpoint(capsys) -> None:
    exit_code = main(
        [
            "diagnose",
            "http://localhost:8000",
            "--intermediate",
            str(FIXTURES / "batch_headroom_middle.prom"),
        ]
    )

    assert exit_code == 2
    assert "only valid with a metrics file" in capsys.readouterr().err


def test_json_report_is_machine_readable(capsys) -> None:
    exit_code = main(["diagnose", str(FIXTURES / "healthy.prom"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == 1
    assert payload["engine"] == "vllm"
    assert payload["sample_count"] == 1
    assert payload["coverage"]["covered_count"] == 2
    assert payload["coverage"]["health_verdict_supported"] is False
    assert payload["findings"][0]["rule_id"] == "INCONCLUSIVE"


def test_fail_on_info_makes_inconclusive_json_actionable(capsys) -> None:
    exit_code = main(
        [
            "diagnose",
            str(FIXTURES / "sparse_vllm.prom"),
            "--previous",
            str(FIXTURES / "sparse_vllm_before.prom"),
            "--interval",
            "10",
            "--json",
            "--fail-on",
            "info",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["findings"][0]["rule_id"] == "INCONCLUSIVE"


def test_complete_healthy_report_clears_strict_threshold(capsys) -> None:
    exit_code = main(
        [
            "diagnose",
            str(FIXTURES / "healthy.prom"),
            "--previous",
            str(FIXTURES / "healthy_before.prom"),
            "--interval",
            "10",
            "--fail-on",
            "info",
        ]
    )

    assert exit_code == 0
    assert "[HEALTHY]" in capsys.readouterr().out


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
        timing=ProbeTiming(
            client_round_trip_ms=125,
            server_accounted_ms=110,
            outside_engine_ms=15,
            outside_engine_ratio=0.12,
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
    assert "Completion HTTP round trip: 125.0ms" in output


def test_nvml_is_rejected_for_offline_fixture(capsys) -> None:
    exit_code = main(["diagnose", str(FIXTURES / "healthy.prom"), "--nvml"])

    assert exit_code == 2
    assert "--nvml is only valid with a live endpoint" in capsys.readouterr().err


def test_live_options_and_api_key_are_forwarded_to_collector(capsys, monkeypatch) -> None:
    observation = collect_files(FIXTURES / "healthy.prom")
    received: dict[str, object] = {}

    def collect(_target: str, **kwargs: object):
        received.update(kwargs)
        return observation

    monkeypatch.setattr("infertop.cli.collect_endpoint", collect)
    monkeypatch.setenv("METRICS_TOKEN", "metrics-secret")

    exit_code = main(
        [
            "diagnose",
            "http://localhost:8000",
            "--nvml",
            "--samples",
            "2",
            "--api-key-env",
            "METRICS_TOKEN",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert received["include_nvml"] is True
    assert received["sample_count"] == 2
    assert received["api_key"] == "metrics-secret"
    assert "Engine: vllm" in output
    assert "metrics-secret" not in output
