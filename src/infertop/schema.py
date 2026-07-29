"""Canonical engine-independent diagnostic schema."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite


def _quantile(buckets: tuple[tuple[float, float], ...], count: float, q: float) -> float | None:
    if count <= 0 or not buckets:
        return None
    target = q * count
    lower_bound = 0.0
    lower_count = 0.0
    for upper_bound, cumulative_count in sorted(buckets):
        if cumulative_count >= target:
            if upper_bound == inf:
                return lower_bound
            bucket_count = cumulative_count - lower_count
            if bucket_count <= 0:
                return upper_bound
            position = (target - lower_count) / bucket_count
            return lower_bound + (upper_bound - lower_bound) * position
        lower_bound = upper_bound
        lower_count = cumulative_count
    return None


@dataclass(frozen=True)
class Distribution:
    """Summary of a Prometheus histogram."""

    count: float
    total: float | None
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    buckets: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_buckets(
        cls,
        buckets: tuple[tuple[float, float], ...],
        *,
        count: float,
        total: float | None,
    ) -> Distribution:
        """Build a summary from cumulative histogram buckets."""

        return cls(
            count=count,
            total=total,
            p50=_quantile(buckets, count, 0.50),
            p90=_quantile(buckets, count, 0.90),
            p95=_quantile(buckets, count, 0.95),
            p99=_quantile(buckets, count, 0.99),
            buckets=tuple(sorted(buckets)),
        )

    def since(self, previous: Distribution | None) -> Distribution:
        """Return the histogram observed since an earlier cumulative scrape."""

        if previous is None or not self.buckets or not previous.buckets:
            return self
        previous_buckets = dict(previous.buckets)
        buckets = tuple(
            (
                bound,
                value - previous_buckets.get(bound, 0.0)
                if value >= previous_buckets.get(bound, 0.0)
                else value,
            )
            for bound, value in self.buckets
        )
        count = self.count - previous.count if self.count >= previous.count else self.count
        total = None
        if self.total is not None and previous.total is not None:
            total = self.total - previous.total if self.total >= previous.total else self.total
        return Distribution.from_buckets(buckets, count=count, total=total)


@dataclass(frozen=True)
class InferenceSnapshot:
    """Normalized state at one point in time."""

    source: str
    captured_at: float
    requests_running: float | None = None
    requests_waiting: float | None = None
    kv_cache_usage: float | None = None
    preemptions_total: float | None = None
    prefix_cache_queries_total: float | None = None
    prefix_cache_hits_total: float | None = None
    prompt_tokens_total: float | None = None
    generation_tokens_total: float | None = None
    end_to_end_latency_seconds: Distribution | None = None
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

    def _counter_delta(self, name: str) -> float | None:
        if self.previous is None:
            return None
        before = getattr(self.previous, name)
        after = getattr(self.current, name)
        if before is None or after is None:
            return None
        # A lower value means the process restarted; the new counter is the delta.
        return after - before if after >= before else after

    def _distribution_since(self, name: str) -> Distribution | None:
        current = getattr(self.current, name)
        if current is None:
            return None
        previous = getattr(self.previous, name) if self.previous is not None else None
        return current.since(previous)

    @property
    def preemptions_delta(self) -> float | None:
        return self._counter_delta("preemptions_total")

    @property
    def preemptions_per_second(self) -> float | None:
        delta = self.preemptions_delta
        interval = self.interval_seconds
        if delta is None or interval is None or interval <= 0 or not isfinite(interval):
            return None
        return delta / interval

    @property
    def prefix_cache_hit_rate(self) -> float | None:
        queries = self._counter_delta("prefix_cache_queries_total")
        hits = self._counter_delta("prefix_cache_hits_total")
        if queries is None or hits is None or queries <= 0:
            return None
        return min(max(hits / queries, 0.0), 1.0)

    @property
    def waiting_is_sustained(self) -> bool:
        if self.previous is None:
            return False
        current = self.current.requests_waiting
        previous = self.previous.requests_waiting
        return current is not None and previous is not None and current > 0 and previous > 0

    @property
    def end_to_end_latency_seconds(self) -> Distribution | None:
        return self._distribution_since("end_to_end_latency_seconds")

    @property
    def queue_latency_seconds(self) -> Distribution | None:
        return self._distribution_since("queue_latency_seconds")

    @property
    def time_to_first_token_seconds(self) -> Distribution | None:
        return self._distribution_since("time_to_first_token_seconds")

    @property
    def time_per_output_token_seconds(self) -> Distribution | None:
        return self._distribution_since("time_per_output_token_seconds")

    @property
    def prompt_tokens(self) -> Distribution | None:
        return self._distribution_since("prompt_tokens")

    @property
    def generation_tokens(self) -> Distribution | None:
        return self._distribution_since("generation_tokens")
