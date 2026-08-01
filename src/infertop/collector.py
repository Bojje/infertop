"""Read-only live and fixture collectors."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from infertop.hardware import collect_nvidia_topology, collect_nvml_gpus
from infertop.normalize import normalize_metrics
from infertop.schema import (
    InferenceObservation,
    InferenceSnapshot,
    validate_tensor_parallel_topology,
)


class CollectionError(RuntimeError):
    """Raised when metrics cannot be collected safely."""


def authorization_headers(api_key: str | None) -> dict[str, str]:
    """Build bearer headers without putting a credential in a URL or exception message."""

    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def metrics_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CollectionError("endpoint must be an http(s) URL")
    path = parsed.path.rstrip("/")
    if not path.endswith("/metrics"):
        path = f"{path}/metrics" if path else "/metrics"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def _scrape(
    client: httpx.Client,
    url: str,
    *,
    include_nvml: bool = False,
) -> InferenceSnapshot:
    response = client.get(url)
    response.raise_for_status()
    snapshot = normalize_metrics(response.text, source=url, captured_at=time.monotonic())
    return replace(snapshot, gpus=collect_nvml_gpus()) if include_nvml else snapshot


def scrape_endpoint(
    endpoint: str,
    *,
    timeout_seconds: float = 5.0,
    include_nvml: bool = False,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> InferenceSnapshot:
    """Read and normalize exactly one metrics scrape."""

    url = metrics_url(endpoint)
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers=authorization_headers(api_key),
            transport=transport,
        ) as client:
            return _scrape(client, url, include_nvml=include_nvml)
    except httpx.HTTPError as exc:
        raise CollectionError(f"could not read {url}: {exc}") from exc


async def scrape_endpoint_async(
    endpoint: str,
    *,
    timeout_seconds: float = 5.0,
    include_nvml: bool = False,
    api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> InferenceSnapshot:
    """Asynchronously read and normalize exactly one metrics scrape."""

    url = metrics_url(endpoint)
    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers=authorization_headers(api_key),
            transport=transport,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            snapshot = normalize_metrics(response.text, source=url, captured_at=time.monotonic())
            return replace(snapshot, gpus=collect_nvml_gpus()) if include_nvml else snapshot
    except httpx.HTTPError as exc:
        raise CollectionError(f"could not read {url}: {exc}") from exc


def collect_endpoint(
    endpoint: str,
    *,
    interval_seconds: float = 1.0,
    timeout_seconds: float = 5.0,
    sample_count: int = 3,
    include_nvml: bool = False,
    api_key: str | None = None,
    tensor_parallel_gpu_indices: tuple[int, ...] = (),
    transport: httpx.BaseTransport | None = None,
) -> InferenceObservation:
    """GET multiple scrapes without redirects or mutation, then normalize them."""

    if interval_seconds <= 0:
        raise CollectionError("interval must be greater than zero")
    if sample_count < 2:
        raise CollectionError("sample_count must be at least two")
    url = metrics_url(endpoint)
    topology = collect_nvidia_topology() if tensor_parallel_gpu_indices else None
    validate_tensor_parallel_topology(topology, tensor_parallel_gpu_indices)
    snapshots = []
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers=authorization_headers(api_key),
            transport=transport,
        ) as client:
            for index in range(sample_count):
                snapshots.append(_scrape(client, url, include_nvml=include_nvml))
                if index < sample_count - 1:
                    time.sleep(interval_seconds)
    except httpx.HTTPError as exc:
        raise CollectionError(f"could not read {url}: {exc}") from exc
    return InferenceObservation(
        current=snapshots[-1],
        previous=snapshots[0],
        intermediate=tuple(snapshots[1:-1]),
        interval_seconds=snapshots[-1].captured_at - snapshots[0].captured_at,
        topology=topology,
        tensor_parallel_gpu_indices=tensor_parallel_gpu_indices,
    )


def collect_files(
    current_path: Path,
    *,
    previous_path: Path | None = None,
    interval_seconds: float | None = None,
) -> InferenceObservation:
    """Load one or two saved metrics scrapes."""

    paths = (previous_path, current_path) if previous_path is not None else (current_path,)
    return collect_file_series(paths, interval_seconds=interval_seconds)


def collect_file_series(
    paths: tuple[Path, ...],
    *,
    interval_seconds: float | None,
) -> InferenceObservation:
    """Load ordered scrapes separated by ``interval_seconds`` for fixture diagnosis."""

    if not paths:
        raise CollectionError("at least one metrics file is required")
    if len(paths) > 1 and (interval_seconds is None or interval_seconds <= 0):
        raise CollectionError("a positive interval is required for multiple metrics files")
    step = interval_seconds if len(paths) > 1 and interval_seconds is not None else 0.0
    snapshots = tuple(
        normalize_metrics(
            path.read_text(),
            source=str(path),
            captured_at=index * step,
        )
        for index, path in enumerate(paths)
    )
    if len(snapshots) == 1:
        return InferenceObservation(current=snapshots[0])
    return InferenceObservation(
        previous=snapshots[0],
        intermediate=snapshots[1:-1],
        current=snapshots[-1],
        interval_seconds=step * (len(snapshots) - 1),
    )
