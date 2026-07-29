from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import Static

from infertop.schema import InferenceSnapshot
from infertop.tui import WatchApp


def _snapshot(index: int) -> InferenceSnapshot:
    return InferenceSnapshot(
        source="http://localhost:8000/metrics",
        captured_at=float(index * 2),
        requests_running=(1, 2, 1)[index],
        requests_waiting=0,
        kv_cache_usage=0.30,
        prompt_tokens_total=100 + index * 50,
        generation_tokens_total=200 + index * 100,
    )


def test_watch_app_renders_engine_findings_without_network() -> None:
    async def exercise() -> None:
        app = WatchApp(
            "http://localhost:8000",
            interval_seconds=60,
            sample_count=3,
            start_polling=False,
        )
        async with app.run_test(size=(100, 40)) as pilot:
            app._accept_snapshot(_snapshot(0))
            app._accept_snapshot(_snapshot(1))
            app._accept_snapshot(_snapshot(2))
            await pilot.pause()

            status = app.query_one("#status", Static)
            report = app.query_one("#report", Static)
            assert "samples 3/3" in str(status.content)
            assert "R5_BATCH_HEADROOM" in str(report.content)
            await pilot.press("q")

    asyncio.run(exercise())


def test_watch_app_polls_with_async_scraper() -> None:
    async def exercise() -> None:
        scrapes = 0
        received: dict[str, object] = {}

        async def scrape(_endpoint: str, **kwargs: object) -> InferenceSnapshot:
            nonlocal scrapes
            received.update(kwargs)
            snapshot = _snapshot(scrapes)
            scrapes += 1
            return snapshot

        app = WatchApp(
            "http://localhost:8000",
            interval_seconds=60,
            sample_count=3,
            include_nvml=True,
            scraper=scrape,
        )
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            assert scrapes == 1
            assert received["include_nvml"] is True
            assert "samples 1/3" in str(app.query_one("#status", Static).content)
            await pilot.press("q")

    asyncio.run(exercise())
