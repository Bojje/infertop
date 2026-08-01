"""Provenance-aware raw metrics capture for fixture development."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from infertop.collector import authorization_headers, metrics_url
from infertop.normalize import NormalizationError, normalize_metrics
from infertop.prometheus import MetricsParseError


class CaptureError(RuntimeError):
    """Raised when a metrics fixture series cannot be captured safely."""


@dataclass(frozen=True)
class CapturedSample:
    file: str
    captured_at: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CaptureManifest:
    schema_version: int
    source: str
    scenario: str
    engine: str
    engine_version: str | None
    model: str | None
    server_command: str | None
    interval_seconds: float
    sample_count: int
    started_at: str
    samples: tuple[CapturedSample, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _public_url(url: str) -> str:
    parsed = urlsplit(url)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(now: Callable[[], datetime]) -> str:
    value = now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _get_metrics(client: httpx.Client, url: str, public_url: str) -> bytes:
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise CaptureError(f"could not read {public_url}: HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise CaptureError(f"could not read {public_url}: {type(exc).__name__}") from exc
    return response.content


def capture_metrics(
    endpoint: str,
    output_directory: Path,
    *,
    scenario: str,
    sample_count: int = 12,
    interval_seconds: float = 1.0,
    timeout_seconds: float = 5.0,
    api_key: str | None = None,
    expected_engine: str | None = None,
    engine_version: str | None = None,
    model: str | None = None,
    server_command: str | None = None,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _utc_now,
) -> CaptureManifest:
    """Capture exact raw scrapes plus a reproducibility manifest into a new directory."""

    if not 2 <= sample_count <= 120:
        raise CaptureError("sample_count must be between 2 and 120")
    if not 0 < interval_seconds <= 60:
        raise CaptureError("interval_seconds must be greater than zero and at most 60")
    if not 0 < timeout_seconds <= 60:
        raise CaptureError("timeout_seconds must be greater than zero and at most 60")
    if not scenario.strip():
        raise CaptureError("scenario must not be empty")
    if expected_engine not in {None, "vllm", "sglang"}:
        raise CaptureError("expected_engine must be vllm or sglang")

    url = metrics_url(endpoint)
    public_url = _public_url(url)
    if output_directory.exists():
        raise CaptureError(f"output directory already exists: {output_directory}")
    started_at = _timestamp(now)
    captured: list[CapturedSample] = []
    detected_engine: str | None = None
    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=False,
        headers=authorization_headers(api_key),
        transport=transport,
    ) as client:
        for index in range(sample_count):
            raw = _get_metrics(client, url, public_url)
            try:
                text = raw.decode("utf-8")
                snapshot = normalize_metrics(text, source=public_url, captured_at=0)
            except (UnicodeDecodeError, MetricsParseError, NormalizationError) as exc:
                raise CaptureError(f"scrape {index} is not supported metrics: {exc}") from exc
            if detected_engine is None:
                detected_engine = snapshot.engine
            elif snapshot.engine != detected_engine:
                raise CaptureError(
                    f"engine changed from {detected_engine} to {snapshot.engine} during capture"
                )
            if expected_engine is not None and snapshot.engine != expected_engine:
                raise CaptureError(
                    f"expected {expected_engine} metrics, detected {snapshot.engine}"
                )
            if index == 0:
                try:
                    output_directory.mkdir(parents=True, exist_ok=False)
                except FileExistsError as exc:
                    raise CaptureError(
                        f"output directory already exists: {output_directory}"
                    ) from exc
            file_name = f"{index:03d}.prom"
            (output_directory / file_name).write_bytes(raw)
            captured.append(
                CapturedSample(
                    file=file_name,
                    captured_at=_timestamp(now),
                    bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
            if index < sample_count - 1:
                sleep(interval_seconds)

    if detected_engine is None:  # pragma: no cover - sample_count validation makes this defensive
        raise CaptureError("capture contained no samples")
    manifest = CaptureManifest(
        schema_version=1,
        source=public_url,
        scenario=scenario,
        engine=detected_engine,
        engine_version=engine_version,
        model=model,
        server_command=server_command,
        interval_seconds=interval_seconds,
        sample_count=sample_count,
        started_at=started_at,
        samples=tuple(captured),
    )
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
