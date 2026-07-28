from __future__ import annotations

import json
from pathlib import Path

from infertop.cli import main

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
    assert "1. CRITICAL [R2_KV_THRASHING]" in output
    assert "KV cache usage: 97.0%" in output
    assert "Preemptions: +7 over 10.0s (0.70/s)" in output


def test_json_report_is_machine_readable(capsys) -> None:
    exit_code = main(["diagnose", str(FIXTURES / "healthy.prom"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == 1
    assert payload["findings"][0]["rule_id"] == "HEALTHY"
