"""Read-only historical input from the Prometheus HTTP API."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from infertop.collector import CollectionError, authorization_headers
from infertop.normalize import normalize_samples, supported_metric_names
from infertop.prometheus import Sample
from infertop.schema import InferenceObservation, InferenceSnapshot

MAX_RANGE_SAMPLES = 120
_LABEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def parse_range_time(value: str) -> float:
    """Parse a Prometheus-compatible Unix or RFC3339 timestamp."""

    try:
        timestamp = float(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(f"{value[:-1]}+00:00" if value.endswith("Z") else value)
        except ValueError as exc:
            raise ValueError("time must be a Unix timestamp or RFC3339 value") from exc
        if parsed.tzinfo is None:
            raise ValueError("RFC3339 time must include a UTC offset or Z suffix") from None
        timestamp = parsed.timestamp()
    if not math.isfinite(timestamp):
        raise ValueError("time must be finite")
    return timestamp


def prometheus_query_url(endpoint: str) -> str:
    """Return the stable range-query endpoint for a Prometheus base URL."""

    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CollectionError("Prometheus endpoint must be an http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise CollectionError("put Prometheus credentials in an environment variable, not the URL")
    if parsed.query:
        raise CollectionError("Prometheus endpoint must not contain query parameters")
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1/query_range"):
        query_path = path
    elif path.endswith("/api/v1"):
        query_path = f"{path}/query_range"
    else:
        query_path = f"{path}/api/v1/query_range" if path else "/api/v1/query_range"
    return urlunsplit((parsed.scheme, parsed.netloc, query_path, "", ""))


def _promql_string(value: str) -> str:
    return value.replace("\\", r"\\").replace("\n", r"\n").replace('"', r"\"")


def build_metric_selector(labels: Mapping[str, str] | None = None) -> str:
    """Build the exact supported-name selector with safely quoted equality matchers."""

    matchers = [
        '__name__=~"^(?:' + "|".join(re.escape(name) for name in supported_metric_names()) + ')$"'
    ]
    label_items = list((labels or {}).items())
    for name, value in label_items:
        if not isinstance(name, str) or not _LABEL_NAME_RE.fullmatch(name) or name == "__name__":
            raise CollectionError(f"invalid Prometheus label name: {name!r}")
        if not isinstance(value, str):
            raise CollectionError(f"Prometheus label {name!r} must have a string value")
    for name, value in sorted(label_items):
        matchers.append(f'{name}="{_promql_string(value)}"')
    return "{" + ",".join(matchers) + "}"


def _matrix_snapshots(payload: Any, *, source: str) -> tuple[InferenceSnapshot, ...]:
    if not isinstance(payload, dict):
        raise CollectionError("Prometheus returned a malformed JSON envelope")
    if payload.get("status") != "success":
        message = payload.get("error")
        detail = message if isinstance(message, str) else "query failed"
        raise CollectionError(f"Prometheus query failed: {detail}")
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("resultType") != "matrix":
        raise CollectionError("Prometheus range query did not return a matrix")
    result = data.get("result")
    if not isinstance(result, list):
        raise CollectionError("Prometheus returned a malformed matrix")

    by_timestamp: dict[float, list[Sample]] = {}
    target_identities: set[tuple[str | None, str | None]] = set()
    for series in result:
        if not isinstance(series, dict):
            raise CollectionError("Prometheus returned a malformed series")
        metric = series.get("metric")
        values = series.get("values")
        if not isinstance(metric, dict) or not isinstance(values, list):
            raise CollectionError("Prometheus returned a malformed range series")
        name = metric.get("__name__")
        if not isinstance(name, str) or name not in supported_metric_names():
            continue
        labels = tuple(
            sorted(
                (label_name, label_value)
                for label_name, label_value in metric.items()
                if label_name != "__name__"
                and isinstance(label_name, str)
                and isinstance(label_value, str)
            )
        )
        job = metric.get("job")
        instance = metric.get("instance")
        if isinstance(job, str) or isinstance(instance, str):
            target_identities.add(
                (
                    job if isinstance(job, str) else None,
                    instance if isinstance(instance, str) else None,
                )
            )
        for point in values:
            if not isinstance(point, list) or len(point) != 2:
                raise CollectionError("Prometheus returned a malformed sample point")
            try:
                timestamp = float(point[0])
                value = float(point[1])
            except (TypeError, ValueError) as exc:
                raise CollectionError("Prometheus returned a non-numeric sample point") from exc
            if not math.isfinite(timestamp) or not math.isfinite(value):
                continue
            by_timestamp.setdefault(timestamp, []).append(
                Sample(name=name, labels=labels, value=value)
            )

    if len(target_identities) > 1:
        raise CollectionError(
            "Prometheus query matched multiple job/instance targets; add --prometheus-label "
            "filters to select one endpoint"
        )
    if len(by_timestamp) < 2:
        raise CollectionError("Prometheus query returned fewer than two usable snapshots")
    if len(by_timestamp) > MAX_RANGE_SAMPLES:
        raise CollectionError(f"Prometheus returned more than {MAX_RANGE_SAMPLES} snapshots")
    return tuple(
        normalize_samples(tuple(by_timestamp[timestamp]), source=source, captured_at=timestamp)
        for timestamp in sorted(by_timestamp)
    )


def collect_prometheus_range(
    endpoint: str,
    *,
    start: float,
    end: float,
    step_seconds: float = 15.0,
    timeout_seconds: float = 5.0,
    labels: Mapping[str, str] | None = None,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> InferenceObservation:
    """Query a bounded historical range and return the canonical observation."""

    if not all(math.isfinite(value) for value in (start, end, step_seconds, timeout_seconds)):
        raise CollectionError("Prometheus range values must be finite")
    if end <= start:
        raise CollectionError("Prometheus --end must be later than --start")
    if step_seconds <= 0:
        raise CollectionError("Prometheus --step must be greater than zero")
    if timeout_seconds <= 0:
        raise CollectionError("timeout must be greater than zero")
    expected_samples = math.floor((end - start) / step_seconds) + 1
    if expected_samples < 2:
        raise CollectionError("Prometheus range must contain at least two samples")
    if expected_samples > MAX_RANGE_SAMPLES:
        raise CollectionError(
            f"Prometheus range is capped at {MAX_RANGE_SAMPLES} samples; increase --step"
        )

    url = prometheus_query_url(endpoint)
    source = f"prometheus:{url.removesuffix('/api/v1/query_range')}"
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers=authorization_headers(api_key),
            transport=transport,
        ) as client:
            response = client.get(
                url,
                params={
                    "query": build_metric_selector(labels),
                    "start": start,
                    "end": end,
                    "step": step_seconds,
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise CollectionError(f"could not query {url}: {exc}") from exc
    except ValueError as exc:
        raise CollectionError(f"Prometheus returned invalid JSON from {url}") from exc

    snapshots = _matrix_snapshots(payload, source=source)
    return InferenceObservation(
        previous=snapshots[0],
        intermediate=snapshots[1:-1],
        current=snapshots[-1],
        interval_seconds=snapshots[-1].captured_at - snapshots[0].captured_at,
    )
