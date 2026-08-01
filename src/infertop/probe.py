"""Opt-in bounded probe for OpenAI-compatible inference endpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

_monotonic_seconds = perf_counter

MAX_PROBE_REQUESTS = 10
MAX_PROBE_OUTPUT_TOKENS_PER_REQUEST = 256
MAX_PROBE_TOTAL_OUTPUT_TOKENS = 1024
MAX_PROBE_PROMPT_CHARACTERS = 32_768


class ProbeError(RuntimeError):
    """Raised when an active inference probe cannot be completed."""


@dataclass(frozen=True)
class RequestMetrics:
    """vLLM per-request timings, available when enabled on the server."""

    time_to_first_token_ms: float | None = None
    generation_time_ms: float | None = None
    queue_time_ms: float | None = None
    mean_itl_ms: float | None = None
    tokens_per_second: float | None = None


@dataclass(frozen=True)
class ProbeTiming:
    """Correlation between the HTTP round trip and server-reported engine phases."""

    client_round_trip_ms: float
    server_accounted_ms: float | None
    outside_engine_ms: float | None
    outside_engine_ratio: float | None


@dataclass(frozen=True)
class ProbeResult:
    endpoint: str
    model: str
    request_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    metrics: RequestMetrics | None
    timing: ProbeTiming
    dominant_phase: str | None
    verdict: str
    evidence: tuple[str, ...]
    remediations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbePercentiles:
    """Nearest-rank summary over the available values in a probe series."""

    count: int
    p50: float | None
    p95: float | None


@dataclass(frozen=True)
class ProbeRunResult:
    """Bounded repeated probes with transparent coverage and percentile summaries."""

    endpoint: str
    model: str
    max_tokens_per_request: int
    results: tuple[ProbeResult, ...]

    @property
    def request_count(self) -> int:
        return len(self.results)

    @property
    def requested_output_token_ceiling(self) -> int:
        return self.request_count * self.max_tokens_per_request

    @property
    def metrics_sample_count(self) -> int:
        return sum(result.metrics is not None for result in self.results)

    @property
    def reported_prompt_tokens(self) -> int | None:
        return _complete_integer_total(self.results, "prompt_tokens")

    @property
    def reported_completion_tokens(self) -> int | None:
        return _complete_integer_total(self.results, "completion_tokens")

    @property
    def dominant_phase_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            name = result.dominant_phase or "unavailable"
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def client_round_trip_ms(self) -> ProbePercentiles:
        return _percentiles(result.timing.client_round_trip_ms for result in self.results)

    def metric_percentiles(self, name: str) -> ProbePercentiles:
        """Summarize one optional server-reported metric across completed probes."""

        return _percentiles(
            value
            for result in self.results
            if result.metrics is not None
            for value in (getattr(result.metrics, name),)
            if value is not None
        )

    @property
    def outside_engine_ms(self) -> ProbePercentiles:
        return _percentiles(
            result.timing.outside_engine_ms
            for result in self.results
            if result.timing.outside_engine_ms is not None
        )

    def to_dict(self) -> dict[str, Any]:
        metric_names = (
            "queue_time_ms",
            "time_to_first_token_ms",
            "generation_time_ms",
            "mean_itl_ms",
            "tokens_per_second",
        )
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "request_count": self.request_count,
            "safety": {
                "request_cap": MAX_PROBE_REQUESTS,
                "max_tokens_per_request": self.max_tokens_per_request,
                "requested_output_token_ceiling": self.requested_output_token_ceiling,
                "total_output_token_cap": MAX_PROBE_TOTAL_OUTPUT_TOKENS,
            },
            "reported_tokens": {
                "prompt": self.reported_prompt_tokens,
                "completion": self.reported_completion_tokens,
            },
            "per_request_metrics_coverage": {
                "available": self.metrics_sample_count,
                "total": self.request_count,
            },
            "percentiles": {
                "client_round_trip_ms": asdict(self.client_round_trip_ms),
                **{name: asdict(self.metric_percentiles(name)) for name in metric_names},
                "outside_engine_ms": asdict(self.outside_engine_ms),
            },
            "dominant_phases": self.dominant_phase_counts,
            "requests": [result.to_dict() for result in self.results],
        }


def _percentiles(values: Any) -> ProbePercentiles:
    known = sorted(float(value) for value in values if value is not None and isfinite(value))
    if not known:
        return ProbePercentiles(count=0, p50=None, p95=None)

    def nearest_rank(quantile: float) -> float:
        return known[max(ceil(quantile * len(known)) - 1, 0)]

    return ProbePercentiles(
        count=len(known),
        p50=nearest_rank(0.50),
        p95=nearest_rank(0.95),
    )


def _complete_integer_total(results: tuple[ProbeResult, ...], name: str) -> int | None:
    values = tuple(getattr(result, name) for result in results)
    return sum(values) if values and all(value is not None for value in values) else None


def api_base_url(endpoint: str) -> str:
    """Normalize a server, metrics, or v1 URL to an OpenAI-compatible API base."""

    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProbeError("endpoint must be an http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ProbeError("put endpoint credentials in an environment variable, not the URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/metrics"):
        path = path[: -len("/metrics")]
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _number(value: Any) -> float | None:
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
    ):
        return float(value)
    return None


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _request_metrics(payload: Any) -> RequestMetrics | None:
    if not isinstance(payload, dict):
        return None
    metrics = RequestMetrics(
        time_to_first_token_ms=_number(payload.get("time_to_first_token_ms")),
        generation_time_ms=_number(payload.get("generation_time_ms")),
        queue_time_ms=_number(payload.get("queue_time_ms")),
        mean_itl_ms=_number(payload.get("mean_itl_ms")),
        tokens_per_second=_number(payload.get("tokens_per_second")),
    )
    return metrics if any(value is not None for value in asdict(metrics).values()) else None


def correlate_probe_timing(
    metrics: RequestMetrics | None,
    client_round_trip_ms: float,
) -> ProbeTiming:
    """Purely compare client time with queue, prefill/TTFT, and decode time."""

    if not isfinite(client_round_trip_ms) or client_round_trip_ms < 0:
        client_round_trip_ms = 0.0
    if metrics is None:
        return ProbeTiming(client_round_trip_ms, None, None, None)
    phases = (
        metrics.queue_time_ms,
        metrics.time_to_first_token_ms,
        metrics.generation_time_ms,
    )
    if any(value is None or not isfinite(value) or value < 0 for value in phases):
        return ProbeTiming(client_round_trip_ms, None, None, None)
    server_accounted_ms = sum(value for value in phases if value is not None)
    outside_engine_ms = max(client_round_trip_ms - server_accounted_ms, 0.0)
    outside_engine_ratio = (
        outside_engine_ms / client_round_trip_ms if client_round_trip_ms > 0 else 0.0
    )
    return ProbeTiming(
        client_round_trip_ms=client_round_trip_ms,
        server_accounted_ms=server_accounted_ms,
        outside_engine_ms=outside_engine_ms,
        outside_engine_ratio=outside_engine_ratio,
    )


_OUTSIDE_ENGINE_MIN_MS = 50.0
_OUTSIDE_ENGINE_MIN_RATIO = 0.20


def _analyze_metrics(
    metrics: RequestMetrics | None,
    timing: ProbeTiming,
) -> tuple[str | None, str, tuple[str, ...], tuple[str, ...]]:
    timing_evidence = [f"Completion HTTP round trip: {timing.client_round_trip_ms:.1f}ms"]
    if timing.server_accounted_ms is not None:
        timing_evidence.append(f"Server-accounted engine time: {timing.server_accounted_ms:.1f}ms")
    if timing.outside_engine_ms is not None and timing.outside_engine_ratio is not None:
        residual = (
            f"Outside engine timing: {timing.outside_engine_ms:.1f}ms "
            f"({timing.outside_engine_ratio:.1%} of round trip)"
        )
        if (
            timing.outside_engine_ms >= _OUTSIDE_ENGINE_MIN_MS
            and timing.outside_engine_ratio >= _OUTSIDE_ENGINE_MIN_RATIO
        ):
            residual += (
                f" (significant: >= {_OUTSIDE_ENGINE_MIN_MS:.0f}ms "
                f"and >= {_OUTSIDE_ENGINE_MIN_RATIO:.0%})"
            )
        timing_evidence.append(residual)
    if metrics is None:
        return (
            None,
            "The request succeeded, but the server returned no per-request timing metrics.",
            (*timing_evidence, "Per-request metrics object: unavailable"),
            ("Start vLLM with --enable-per-request-metrics for phase-level evidence.",),
        )
    phases = {
        "queue": metrics.queue_time_ms,
        "prefill/TTFT": metrics.time_to_first_token_ms,
        "decode": metrics.generation_time_ms,
    }
    available = {name: value for name, value in phases.items() if value is not None}
    evidence = tuple(
        [
            *timing_evidence,
            *(
                f"{name}: {value:.1f}ms"
                for name, value in (
                    ("Queue", metrics.queue_time_ms),
                    ("TTFT after scheduling", metrics.time_to_first_token_ms),
                    ("Decode", metrics.generation_time_ms),
                    ("Mean ITL", metrics.mean_itl_ms),
                )
                if value is not None
            ),
            *(
                (f"Output throughput: {metrics.tokens_per_second:.1f} tokens/s",)
                if metrics.tokens_per_second is not None
                else ()
            ),
        ]
    )
    outside_engine_is_significant = (
        timing.outside_engine_ms is not None
        and timing.outside_engine_ratio is not None
        and timing.outside_engine_ms >= _OUTSIDE_ENGINE_MIN_MS
        and timing.outside_engine_ratio >= _OUTSIDE_ENGINE_MIN_RATIO
    )
    if outside_engine_is_significant:
        available["outside engine"] = timing.outside_engine_ms
    if not available:
        return (
            None,
            "The metrics object contained no usable phase timings.",
            evidence,
            ("Check the vLLM per-request metrics configuration.",),
        )
    dominant_phase = max(available, key=available.__getitem__)
    if dominant_phase == "outside engine":
        verdict = (
            "Client-observed latency is dominated by time outside the server-reported "
            "queue, prefill, and decode phases."
        )
        remediations = (
            "Repeat the probe from the inference host to separate network and proxy overhead.",
            (
                "Inspect reverse proxies, API middleware, response serialization, "
                "and client-side buffering."
            ),
            "Treat the residual as unattributed time, not proof of a specific network fault.",
        )
    elif dominant_phase == "queue":
        verdict = "This probe spent most of its measured time waiting in the scheduler queue."
        remediations = ("Run diagnose to confirm sustained saturation before scaling replicas.",)
    elif dominant_phase == "prefill/TTFT":
        verdict = "This probe spent most of its measured time reaching the first token."
        remediations = ("Inspect prompt length and prefix-cache reuse with diagnose.",)
    else:
        verdict = "This probe spent most of its measured time decoding output tokens."
        remediations = (
            "Retest with a representative output length before evaluating speculative decoding.",
        )
    return dominant_phase, verdict, evidence, remediations


def _validate_probe_options(*, prompt: str, max_tokens: int, timeout_seconds: float) -> None:
    if not 1 <= max_tokens <= MAX_PROBE_OUTPUT_TOKENS_PER_REQUEST:
        raise ProbeError(f"max_tokens must be between 1 and {MAX_PROBE_OUTPUT_TOKENS_PER_REQUEST}")
    if not 1 <= len(prompt) <= MAX_PROBE_PROMPT_CHARACTERS:
        raise ProbeError(
            f"prompt must contain between 1 and {MAX_PROBE_PROMPT_CHARACTERS} characters"
        )
    if not 0 < timeout_seconds <= 300:
        raise ProbeError("timeout_seconds must be greater than zero and at most 300")


def _discover_model(client: httpx.Client, base_url: str) -> str:
    models_response = client.get(f"{base_url}/models")
    models_response.raise_for_status()
    models_payload = models_response.json()
    data = models_payload.get("data") if isinstance(models_payload, dict) else None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ProbeError("/v1/models returned no usable model")
    model_id = data[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ProbeError("/v1/models returned a model without an id")
    return model_id


def _probe_with_client(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> ProbeResult:
    request_started = _monotonic_seconds()
    response = client.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
    )
    client_round_trip_ms = (_monotonic_seconds() - request_started) * 1000
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ProbeError("chat completion returned a non-object response")
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    metrics = _request_metrics(payload.get("metrics"))
    timing = correlate_probe_timing(metrics, client_round_trip_ms)
    dominant_phase, verdict, evidence, remediations = _analyze_metrics(metrics, timing)
    request_id = payload.get("id")
    return ProbeResult(
        endpoint=base_url,
        model=model,
        request_id=request_id if isinstance(request_id, str) else None,
        prompt_tokens=_integer(usage.get("prompt_tokens")),
        completion_tokens=_integer(usage.get("completion_tokens")),
        metrics=metrics,
        timing=timing,
        dominant_phase=dominant_phase,
        verdict=verdict,
        evidence=evidence,
        remediations=remediations,
    )


def probe_endpoint(
    endpoint: str,
    *,
    model: str | None = None,
    prompt: str = "Reply with exactly the word OK.",
    max_tokens: int = 8,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> ProbeResult:
    """Send one bounded, non-streaming completion and parse per-request metrics."""

    _validate_probe_options(
        prompt=prompt,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    base_url = api_base_url(endpoint)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers=headers,
            transport=transport,
        ) as client:
            selected_model = model or _discover_model(client, base_url)
            return _probe_with_client(
                client,
                base_url=base_url,
                model=selected_model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
    except ProbeError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise ProbeError(f"active probe failed for {base_url}: {exc}") from exc


def probe_endpoint_repeated(
    endpoint: str,
    *,
    request_count: int,
    model: str | None = None,
    prompt: str = "Reply with exactly the word OK.",
    max_tokens: int = 8,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> ProbeRunResult:
    """Send a small sequential series after enforcing request and token ceilings."""

    if not 1 <= request_count <= MAX_PROBE_REQUESTS:
        raise ProbeError(f"request_count must be between 1 and {MAX_PROBE_REQUESTS}")
    requested_tokens = request_count * max_tokens
    if requested_tokens > MAX_PROBE_TOTAL_OUTPUT_TOKENS:
        raise ProbeError(
            "repeated probe output ceiling exceeds "
            f"{MAX_PROBE_TOTAL_OUTPUT_TOKENS} tokens; lower request_count or max_tokens"
        )

    _validate_probe_options(
        prompt=prompt,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    base_url = api_base_url(endpoint)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers=headers,
            transport=transport,
        ) as client:
            selected_model = model or _discover_model(client, base_url)
            results = tuple(
                _probe_with_client(
                    client,
                    base_url=base_url,
                    model=selected_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                for _index in range(request_count)
            )
    except ProbeError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise ProbeError(f"active probe failed for {base_url}: {exc}") from exc
    return ProbeRunResult(
        endpoint=base_url,
        model=selected_model,
        max_tokens_per_request=max_tokens,
        results=results,
    )
