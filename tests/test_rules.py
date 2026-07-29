from __future__ import annotations

from pathlib import Path

import pytest

from infertop.collector import collect_file_series, collect_files
from infertop.rules import (
    diagnose,
    rule_batch_efficiency,
    rule_hardware_correlation,
    rule_kv_cache_health,
)
from infertop.schema import (
    Distribution,
    GpuDeviceSnapshot,
    InferenceObservation,
    InferenceSnapshot,
)

FIXTURES = Path(__file__).parent / "fixtures"
SCENARIOS = (
    "healthy",
    "queue_saturated",
    "kv_thrashing",
    "prefill_bound",
    "decode_bound",
    "sglang_healthy",
    "sglang_thrashing",
)


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


def test_counter_delta_detects_reset_inside_multi_sample_window() -> None:
    observation = InferenceObservation(
        previous=InferenceSnapshot(source="fixture", captured_at=0, preemptions_total=100),
        intermediate=(InferenceSnapshot(source="fixture", captured_at=5, preemptions_total=2),),
        current=InferenceSnapshot(source="fixture", captured_at=10, preemptions_total=150),
        interval_seconds=10,
    )

    assert observation.preemptions_delta == 150
    assert observation.preemptions_per_second == pytest.approx(15)


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


@pytest.mark.parametrize(
    ("scenario", "expected"),
    (
        ("batch_headroom", "R5_BATCH_HEADROOM"),
        ("concurrency_ceiling", "R5_CONCURRENCY_CEILING"),
    ),
)
def test_r5_fixture_golden_top_finding(scenario: str, expected: str) -> None:
    observation = collect_file_series(
        (
            FIXTURES / f"{scenario}_before.prom",
            FIXTURES / f"{scenario}_middle.prom",
            FIXTURES / f"{scenario}.prom",
        ),
        interval_seconds=10,
    )

    golden = (FIXTURES / f"{scenario}.top").read_text().strip()
    assert golden == expected
    assert diagnose(observation)[0].rule_id == golden


def test_r5_requires_three_samples() -> None:
    observation = collect_files(
        FIXTURES / "batch_headroom.prom",
        previous_path=FIXTURES / "batch_headroom_before.prom",
        interval_seconds=10,
    )

    assert rule_batch_efficiency(observation) is None


def test_sglang_thrashing_uses_engine_specific_evidence_and_remediation() -> None:
    observation = collect_files(
        FIXTURES / "sglang_thrashing.prom",
        previous_path=FIXTURES / "sglang_thrashing_before.prom",
        interval_seconds=10,
    )

    finding = diagnose(observation)[0]

    assert finding.rule_id == "R3_KV_THRASHING"
    assert any("Retractions: +4" in item for item in finding.evidence)
    assert any("--schedule-conservativeness" in item for item in finding.remediations)
    assert all("--max-num-seqs" not in item for item in finding.remediations)


def _latency_distribution(p95: float) -> Distribution:
    return Distribution(
        count=10,
        total=None,
        p50=p95 / 2,
        p90=p95,
        p95=p95,
        p99=p95,
    )


def _hardware_observation(
    *,
    gpu_utilization: float,
    memory_io_utilization: float,
    e2e_p95: float | None = None,
    ttft_p95: float | None = None,
    itl_p95: float | None = None,
) -> InferenceObservation:
    gpu = GpuDeviceSnapshot(
        index=0,
        name="NVIDIA GeForce RTX 5080",
        uuid="GPU-test",
        gpu_utilization=gpu_utilization,
        memory_io_utilization=memory_io_utilization,
        memory_used_bytes=12,
        memory_total_bytes=16,
        power_watts=270,
        power_limit_watts=360,
    )
    return InferenceObservation(
        previous=InferenceSnapshot(
            source="fixture",
            captured_at=0,
            engine="vllm",
            gpus=(gpu,),
            requests_running=2,
        ),
        current=InferenceSnapshot(
            source="fixture",
            captured_at=10,
            engine="vllm",
            gpus=(gpu,),
            requests_running=2,
            end_to_end_latency_seconds=(
                _latency_distribution(e2e_p95) if e2e_p95 is not None else None
            ),
            time_to_first_token_seconds=(
                _latency_distribution(ttft_p95) if ttft_p95 is not None else None
            ),
            time_per_output_token_seconds=(
                _latency_distribution(itl_p95) if itl_p95 is not None else None
            ),
        ),
        interval_seconds=10,
    )


def test_r6_correlates_high_ttft_with_sustained_compute_pressure() -> None:
    finding = rule_hardware_correlation(
        _hardware_observation(
            gpu_utilization=0.95,
            memory_io_utilization=0.60,
            ttft_p95=1.5,
        )
    )

    assert finding is not None
    assert finding.rule_id == "R6_COMPUTE_PRESSURE"


def test_r6_correlates_high_itl_with_sustained_device_memory_activity() -> None:
    finding = rule_hardware_correlation(
        _hardware_observation(
            gpu_utilization=0.75,
            memory_io_utilization=0.96,
            itl_p95=0.2,
        )
    )

    assert finding is not None
    assert finding.rule_id == "R6_DEVICE_MEMORY_ACTIVITY"
    assert "does not prove" in finding.summary


def test_r6_flags_high_latency_while_active_local_gpu_is_idle() -> None:
    finding = rule_hardware_correlation(
        _hardware_observation(
            gpu_utilization=0.10,
            memory_io_utilization=0.15,
            e2e_p95=3.0,
        )
    )

    assert finding is not None
    assert finding.rule_id == "R6_LOW_GPU_ACTIVITY"
