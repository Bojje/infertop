from __future__ import annotations

from pathlib import Path

import pytest

from infertop.collector import collect_files
from infertop.rules import diagnose, rule_kv_thrashing
from infertop.schema import InferenceObservation, InferenceSnapshot

FIXTURES = Path(__file__).parent / "fixtures"
SCENARIOS = ("healthy", "queue_saturated", "kv_thrashing", "prefill_bound", "decode_bound")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_golden_top_finding(scenario: str) -> None:
    observation = collect_files(
        FIXTURES / f"{scenario}.prom",
        previous_path=FIXTURES / f"{scenario}_before.prom",
        interval_seconds=10,
    )

    expected = (FIXTURES / f"{scenario}.top").read_text().strip()
    assert diagnose(observation)[0].rule_id == expected


def test_kv_rule_does_not_claim_thrashing_from_one_lifetime_counter() -> None:
    observation = InferenceObservation(
        current=InferenceSnapshot(
            source="fixture",
            captured_at=0,
            kv_cache_usage=0.99,
            preemptions_total=999,
        )
    )

    assert rule_kv_thrashing(observation) is None


def test_counter_reset_uses_new_counter_value_as_delta() -> None:
    observation = InferenceObservation(
        previous=InferenceSnapshot(source="fixture", captured_at=0, preemptions_total=100),
        current=InferenceSnapshot(source="fixture", captured_at=10, preemptions_total=3),
        interval_seconds=10,
    )

    assert observation.preemptions_delta == 3
    assert observation.preemptions_per_second == pytest.approx(0.3)
