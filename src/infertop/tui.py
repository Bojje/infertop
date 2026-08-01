"""Minimal Textual watch view backed by the deterministic diagnosis engine."""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Static

from infertop.collector import CollectionError, scrape_endpoint_async
from infertop.hardware import collect_nvidia_topology
from infertop.report import render_text
from infertop.rules import diagnose
from infertop.schema import (
    InferenceObservation,
    InferenceSnapshot,
    validate_tensor_parallel_topology,
)

Scraper = Callable[..., Awaitable[InferenceSnapshot]]


class WatchApp(App[None]):
    """Continuously scrape, diagnose, and render the ranked report."""

    TITLE = "infertop"
    BINDINGS: ClassVar = [
        ("q", "quit", "Quit"),
        ("r", "refresh_now", "Refresh"),
    ]
    CSS = """
    Screen {
        background: $surface;
    }
    #status {
        height: 3;
        padding: 1 2;
        background: $boost;
        color: $text;
    }
    #report-scroll {
        height: 1fr;
        padding: 1 2;
    }
    #report {
        width: 100%;
        height: auto;
    }
    """

    def __init__(
        self,
        endpoint: str,
        *,
        interval_seconds: float = 2.0,
        timeout_seconds: float = 5.0,
        sample_count: int = 3,
        include_nvml: bool = False,
        api_key: str | None = None,
        tensor_parallel_gpu_indices: tuple[int, ...] = (),
        scraper: Scraper = scrape_endpoint_async,
        start_polling: bool = True,
    ) -> None:
        if interval_seconds < 0.25:
            raise ValueError("watch interval must be at least 0.25 seconds")
        if sample_count < 2:
            raise ValueError("watch sample_count must be at least two")
        super().__init__()
        self.endpoint = endpoint
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.sample_count = sample_count
        self.include_nvml = include_nvml
        self.api_key = api_key
        self.tensor_parallel_gpu_indices = tensor_parallel_gpu_indices
        self.topology = collect_nvidia_topology() if self.tensor_parallel_gpu_indices else None
        validate_tensor_parallel_topology(self.topology, self.tensor_parallel_gpu_indices)
        self._scraper = scraper
        self._start_polling = start_polling
        self._snapshots: deque[InferenceSnapshot] = deque(maxlen=sample_count)
        self.sub_title = endpoint

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Connecting to metrics endpoint...", id="status", markup=False)
        with VerticalScroll(id="report-scroll"):
            yield Static("", id="report", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        if not self._start_polling:
            return
        self.poll_metrics()
        self.set_interval(self.interval_seconds, self.poll_metrics)

    def action_refresh_now(self) -> None:
        self.poll_metrics()

    @work(exclusive=True, group="metrics", exit_on_error=False)
    async def poll_metrics(self) -> None:
        try:
            snapshot = await self._scraper(
                self.endpoint,
                timeout_seconds=self.timeout_seconds,
                include_nvml=self.include_nvml,
                api_key=self.api_key,
            )
        except (CollectionError, OSError, RuntimeError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self._accept_snapshot(snapshot)

    def _show_error(self, message: str) -> None:
        self.query_one("#status", Static).update(f"ERROR  {message}")

    def _accept_snapshot(self, snapshot: InferenceSnapshot) -> None:
        self._snapshots.append(snapshot)
        snapshots = tuple(self._snapshots)
        if len(snapshots) == 1:
            observation = InferenceObservation(
                current=snapshot,
                topology=self.topology,
                tensor_parallel_gpu_indices=self.tensor_parallel_gpu_indices,
            )
        else:
            observation = InferenceObservation(
                previous=snapshots[0],
                intermediate=snapshots[1:-1],
                current=snapshots[-1],
                interval_seconds=snapshots[-1].captured_at - snapshots[0].captured_at,
                topology=self.topology,
                tensor_parallel_gpu_indices=self.tensor_parallel_gpu_indices,
            )
        findings = diagnose(observation)
        current = observation.current
        total_tps = observation.total_tokens_per_second
        status_parts = [
            f"samples {observation.sample_count}/{self.sample_count}",
            (
                f"running {current.requests_running:g}"
                if current.requests_running is not None
                else "running ?"
            ),
            (
                f"waiting {current.requests_waiting:g}"
                if current.requests_waiting is not None
                else "waiting ?"
            ),
            (f"KV {current.kv_cache_usage:.1%}" if current.kv_cache_usage is not None else "KV ?"),
            f"tokens {total_tps:.1f}/s" if total_tps is not None else "tokens warming up",
        ]
        self.query_one("#status", Static).update("  |  ".join(status_parts))
        self.query_one("#report", Static).update(render_text(observation, findings))


def run_watch(
    endpoint: str,
    *,
    interval_seconds: float = 2.0,
    timeout_seconds: float = 5.0,
    sample_count: int = 3,
    include_nvml: bool = False,
    api_key: str | None = None,
    tensor_parallel_gpu_indices: tuple[int, ...] = (),
) -> None:
    WatchApp(
        endpoint,
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
        sample_count=sample_count,
        include_nvml=include_nvml,
        api_key=api_key,
        tensor_parallel_gpu_indices=tensor_parallel_gpu_indices,
    ).run()
