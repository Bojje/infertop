from __future__ import annotations

from pathlib import Path

import pytest

from infertop.collector import collect_files
from infertop.rules import diagnose, rule_kv_cache_health
from infertop.schema import Distribution, InferenceObservation, InferenceSnapshot

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

    finding = rule_kv_cache_health(observation)
    assert finding is not None
    assert finding.rule_id == "R3_KV_PRESSURE"


def test_counter_reset_uses_new_counter_value_as_delta() -> None:
    observation = InferenceObservation(
        previous=InferenceSnapshot(source="fixture", captured_at=0, preemptions_total=100),
        current=InferenceSnapshot(source="fixture", captured_at=10, preemptions_total=3),
        interval_seconds=10,
    )

    assert observation.preemptions_delta == 3
    assert observation.preemptions_per_second == pytest.approx(0.3)


def test_histogram_quantiles_use_only_samples_between_scrapes() -> None:
    previous_distribution = Distribution.from_buckets(
        ((1.0, 90), (10.0, 100), (float("inf"), 100)),
        count=100,
        total=180,
    )
    current_distribution = Distribution.from_buckets(
        ((1.0, 90), (10.0, 110), (float("inf"), 110)),
        count=110,
        total=280,
    )
    observation = InferenceObservation(
        previous=InferenceSnapshot(
            source="fixture",
            captured_at=0,
            time_to_first_token_seconds=previous_distribution,
        ),
        current=InferenceSnapshot(
            source="fixture",
            captured_at=10,
            time_to_first_token_seconds=current_distribution,
        ),
        interval_seconds=10,
    )

    assert observation.time_to_first_token_seconds is not None
    assert observation.time_to_first_token_seconds.p50 == pytest.approx(5.5)
    assert observation.time_to_first_token_seconds.p95 == pytest.approx(9.55)


def test_prefix_cache_hit_rate_handles_alias_normalized_counter_deltas() -> None:
    observation = InferenceObservation(
        previous=InferenceSnapshot(
            source="fixture",
            captured_at=0,
            prefix_cache_queries_total=100,
            prefix_cache_hits_total=25,
        ),
        current=InferenceSnapshot(
            source="fixture",
            captured_at=10,
            prefix_cache_queries_total=200,
            prefix_cache_hits_total=55,
        ),
        interval_seconds=10,
    )

    assert observation.prefix_cache_hit_rate == pytest.approx(0.30)
