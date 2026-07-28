"""Canonical engine-independent diagnostic schema."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Distribution:
    """Summary of a Prometheus histogram."""

    count: float
    total: float | None
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None


@dataclass(frozen=True)
class InferenceSnapshot:
    """Normalized state at one point in time."""

    source: str
    captured_at: float
    requests_running: float | None = None
    requests_waiting: float | None = None
    kv_cache_usage: float | None = None
    preemptions_total: float | None = None
    queue_latency_seconds: Distribution | None = None
    time_to_first_token_seconds: Distribution | None = None
    time_per_output_token_seconds: Distribution | None = None
    prompt_tokens: Distribution | None = None
    generation_tokens: Distribution | None = None
    prefill_time_seconds: Distribution | None = None
    decode_time_seconds: Distribution | None = None


@dataclass(frozen=True)
class InferenceObservation:
    """One snapshot, optionally paired with an earlier one for rate evidence."""

    current: InferenceSnapshot
    previous: InferenceSnapshot | None = None
    interval_seconds: float | None = None

    @property
    def preemptions_delta(self) -> float | None:
        if self.previous is None:
            return None
        before = self.previous.preemptions_total
        after = self.current.preemptions_total
        if before is None or after is None:
            return None
        # A lower value means the process restarted; the new counter is the delta.
        return after - before if after >= before else after

    @property
    def preemptions_per_second(self) -> float | None:
        delta = self.preemptions_delta
        interval = self.interval_seconds
        if delta is None or interval is None or interval <= 0 or not isfinite(interval):
            return None
        return delta / interval
