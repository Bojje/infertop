"""Read-only live and fixture collectors."""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from infertop.normalize import normalize_vllm
from infertop.schema import InferenceObservation


class CollectionError(RuntimeError):
    """Raised when metrics cannot be collected safely."""


def metrics_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CollectionError("endpoint must be an http(s) URL")
    path = parsed.path.rstrip("/")
    if not path.endswith("/metrics"):
        path = f"{path}/metrics" if path else "/metrics"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def collect_endpoint(
    endpoint: str,
    *,
    interval_seconds: float = 1.0,
    timeout_seconds: float = 5.0,
) -> InferenceObservation:
    """GET two scrapes without redirects or mutation, then normalize them."""

    if interval_seconds <= 0:
        raise CollectionError("interval must be greater than zero")
    url = metrics_url(endpoint)
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=False) as client:
            first_response = client.get(url)
            first_response.raise_for_status()
            first_time = time.monotonic()
            time.sleep(interval_seconds)
            second_response = client.get(url)
            second_response.raise_for_status()
            second_time = time.monotonic()
    except httpx.HTTPError as exc:
        raise CollectionError(f"could not read {url}: {exc}") from exc
    return InferenceObservation(
        current=normalize_vllm(second_response.text, source=url, captured_at=second_time),
        previous=normalize_vllm(first_response.text, source=url, captured_at=first_time),
        interval_seconds=second_time - first_time,
    )


def collect_files(
    current_path: Path,
    *,
    previous_path: Path | None = None,
    interval_seconds: float | None = None,
) -> InferenceObservation:
    """Load one or two saved metrics scrapes."""

    current = normalize_vllm(current_path.read_text(), source=str(current_path), captured_at=1.0)
    previous = None
    if previous_path is not None:
        previous = normalize_vllm(
            previous_path.read_text(),
            source=str(previous_path),
            captured_at=0.0,
        )
    return InferenceObservation(
        current=current,
        previous=previous,
        interval_seconds=interval_seconds if previous is not None else None,
    )
