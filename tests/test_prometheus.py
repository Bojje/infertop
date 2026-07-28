from __future__ import annotations

import pytest

from infertop.prometheus import MetricsParseError, parse_metrics


def test_parses_colon_names_labels_scientific_values_and_comments() -> None:
    samples = parse_metrics(
        """
        # HELP vllm:num_requests_waiting waiting
        vllm:num_requests_waiting{model_name="Qwen/Qwen3-0.6B",worker="a\\\"b"} 2e1
        """
    )

    assert len(samples) == 1
    assert samples[0].name == "vllm:num_requests_waiting"
    assert samples[0].value == 20
    assert samples[0].label("worker") == 'a"b'


def test_rejects_malformed_sample() -> None:
    with pytest.raises(MetricsParseError, match="line 1"):
        parse_metrics("this is not exposition")
