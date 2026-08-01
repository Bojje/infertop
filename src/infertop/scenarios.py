"""Bounded traffic scenarios used to generate diagnostic fixtures."""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any

import httpx

from infertop.probe import ProbeError, api_base_url

MAX_REQUESTS = 128
MAX_CONCURRENCY = 32
MIN_PROMPT_WORDS = 12
MAX_PROMPT_WORDS = 8192
MAX_OUTPUT_TOKENS = 256
MAX_LAUNCH_INTERVAL_SECONDS = 5.0
DEMO_SCENARIO_NAME = "demo-transition"
MAX_DEMO_REQUESTS = 128
MAX_DEMO_OUTPUT_TOKENS = 32_768

_async_sleep = asyncio.sleep


class ScenarioError(RuntimeError):
    """Raised when a bounded load scenario is invalid or cannot start."""


@dataclass(frozen=True)
class Scenario:
    """One deterministic traffic shape, with hard bounds enforced before execution."""

    name: str
    description: str
    request_count: int
    concurrency: int
    prompt_words: int
    max_tokens: int


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            name="healthy",
            description="low-concurrency short prompts and short outputs",
            request_count=8,
            concurrency=1,
            prompt_words=32,
            max_tokens=32,
        ),
        Scenario(
            name="queue-saturated",
            description="a burst whose concurrency is intended to build a scheduler queue",
            request_count=64,
            concurrency=16,
            prompt_words=128,
            max_tokens=64,
        ),
        Scenario(
            name="kv-pressure",
            description="concurrent long contexts and outputs intended to consume KV capacity",
            request_count=64,
            concurrency=32,
            prompt_words=8192,
            max_tokens=256,
        ),
        Scenario(
            name="prefill-bound",
            description="RAG-like long prompts with very short outputs",
            request_count=12,
            concurrency=4,
            prompt_words=4096,
            max_tokens=8,
        ),
        Scenario(
            name="decode-bound",
            description="short prompts with long outputs",
            request_count=16,
            concurrency=4,
            prompt_words=32,
            max_tokens=256,
        ),
    )
}


@dataclass(frozen=True)
class DemoStage:
    """One fixed stage in the deterministic watch demonstration."""

    name: str
    description: str
    scenario: Scenario
    launch_interval_seconds: float = 0.0


DEMO_STAGES = (
    DemoStage(
        name="healthy-baseline",
        description="paced low-concurrency requests establish a healthy baseline",
        scenario=replace(SCENARIOS["healthy"], name="demo-healthy-baseline"),
        launch_interval_seconds=1.0,
    ),
    DemoStage(
        name="queue-pressure",
        description="one bounded long-output burst creates scheduler pressure",
        scenario=replace(
            SCENARIOS["queue-saturated"],
            name="demo-queue-pressure",
            max_tokens=256,
        ),
    ),
    DemoStage(
        name="healthy-recovery",
        description="paced low-concurrency requests show recovery after the burst",
        scenario=replace(SCENARIOS["healthy"], name="demo-healthy-recovery"),
        launch_interval_seconds=1.0,
    ),
)


@dataclass(frozen=True)
class ScenarioRequestResult:
    index: int
    succeeded: bool
    status_code: int | None
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class ScenarioRunResult:
    endpoint: str
    model: str
    scenario: Scenario
    requests: tuple[ScenarioRequestResult, ...]

    @property
    def succeeded(self) -> int:
        return sum(result.succeeded for result in self.requests)

    @property
    def failed(self) -> int:
        return len(self.requests) - self.succeeded

    @property
    def p50_latency_ms(self) -> float | None:
        return _quantile(self.requests, 0.50)

    @property
    def p95_latency_ms(self) -> float | None:
        return _quantile(self.requests, 0.95)

    @property
    def prompt_tokens(self) -> int | None:
        return _token_total(self.requests, "prompt_tokens")

    @property
    def completion_tokens(self) -> int | None:
        return _token_total(self.requests, "completion_tokens")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            succeeded=self.succeeded,
            failed=self.failed,
            p50_latency_ms=self.p50_latency_ms,
            p95_latency_ms=self.p95_latency_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )
        return payload


@dataclass(frozen=True)
class DemoStageResult:
    stage: DemoStage
    result: ScenarioRunResult


@dataclass(frozen=True)
class DemoRunResult:
    """Result of the fixed healthy-pressure-recovery demonstration."""

    endpoint: str
    model: str
    stages: tuple[DemoStageResult, ...]

    @property
    def request_count(self) -> int:
        return sum(stage.result.scenario.request_count for stage in self.stages)

    @property
    def requested_output_token_ceiling(self) -> int:
        return sum(
            stage.result.scenario.request_count * stage.result.scenario.max_tokens
            for stage in self.stages
        )

    @property
    def succeeded(self) -> int:
        return sum(stage.result.succeeded for stage in self.stages)

    @property
    def failed(self) -> int:
        return sum(stage.result.failed for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "safety": {
                "request_count": self.request_count,
                "request_cap": MAX_DEMO_REQUESTS,
                "requested_output_token_ceiling": self.requested_output_token_ceiling,
                "output_token_cap": MAX_DEMO_OUTPUT_TOKENS,
            },
            "succeeded": self.succeeded,
            "failed": self.failed,
            "stages": [
                {
                    "name": stage.stage.name,
                    "description": stage.stage.description,
                    "launch_interval_seconds": stage.stage.launch_interval_seconds,
                    "result": stage.result.to_dict(),
                }
                for stage in self.stages
            ],
        }


def _quantile(results: tuple[ScenarioRequestResult, ...], quantile: float) -> float | None:
    values = sorted(result.latency_ms for result in results if result.succeeded)
    if not values:
        return None
    index = max(math.ceil(quantile * len(values)) - 1, 0)
    return values[index]


def _token_total(
    results: tuple[ScenarioRequestResult, ...],
    field: str,
) -> int | None:
    values = [getattr(result, field) for result in results]
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def configured_scenario(
    name: str,
    *,
    request_count: int | None = None,
    concurrency: int | None = None,
    prompt_words: int | None = None,
    max_tokens: int | None = None,
) -> Scenario:
    """Return a validated preset with optional bounded overrides."""

    try:
        scenario = SCENARIOS[name]
    except KeyError as exc:
        raise ScenarioError(f"unknown scenario: {name}") from exc
    scenario = replace(
        scenario,
        request_count=scenario.request_count if request_count is None else request_count,
        concurrency=scenario.concurrency if concurrency is None else concurrency,
        prompt_words=scenario.prompt_words if prompt_words is None else prompt_words,
        max_tokens=scenario.max_tokens if max_tokens is None else max_tokens,
    )
    _validate_scenario(scenario)
    return scenario


def _validate_scenario(scenario: Scenario) -> None:
    bounds = (
        ("request_count", scenario.request_count, 1, MAX_REQUESTS),
        ("concurrency", scenario.concurrency, 1, MAX_CONCURRENCY),
        ("prompt_words", scenario.prompt_words, MIN_PROMPT_WORDS, MAX_PROMPT_WORDS),
        ("max_tokens", scenario.max_tokens, 1, MAX_OUTPUT_TOKENS),
    )
    for name, value, minimum, maximum in bounds:
        if not minimum <= value <= maximum:
            raise ScenarioError(f"{name} must be between {minimum} and {maximum}")
    if scenario.concurrency > scenario.request_count:
        raise ScenarioError("concurrency cannot exceed request_count")


def _prompt(scenario: Scenario, index: int) -> str:
    # Put the request-specific marker in the first cache block so prefix caching cannot turn a
    # long-prompt scenario into an accidental cache-hit benchmark.
    marker = f"request-{index}-for-{scenario.name}."
    if scenario.max_tokens <= 8:
        instruction = "After reading the context, answer with only OK."
    else:
        instruction = (
            f"Now output the word token exactly {scenario.max_tokens} times without commentary."
        )
    fixed = [marker, *instruction.split()]
    context = ["measurement"] * (scenario.prompt_words - len(fixed))
    return " ".join((marker, *context, *instruction.split()))


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _safe_failure(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


async def _discover_model(client: httpx.AsyncClient, base_url: str) -> str:
    try:
        response = await client.get(f"{base_url}/models")
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ScenarioError(f"could not discover a model: {_safe_failure(exc)}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ScenarioError("/v1/models returned no usable model")
    model = data[0].get("id")
    if not isinstance(model, str) or not model:
        raise ScenarioError("/v1/models returned a model without an id")
    return model


async def run_scenario(
    endpoint: str,
    scenario: Scenario,
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 120.0,
    launch_interval_seconds: float = 0.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ScenarioRunResult:
    """Execute one explicitly requested, bounded OpenAI-compatible traffic scenario."""

    _validate_scenario(scenario)
    if not 0 < timeout_seconds <= 300:
        raise ScenarioError("timeout_seconds must be greater than zero and at most 300")
    if not 0 <= launch_interval_seconds <= MAX_LAUNCH_INTERVAL_SECONDS:
        raise ScenarioError(
            f"launch_interval_seconds must be between zero and {MAX_LAUNCH_INTERVAL_SECONDS:g}"
        )
    try:
        base_url = api_base_url(endpoint)
    except ProbeError as exc:
        raise ScenarioError(str(exc)) from exc
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    semaphore = asyncio.Semaphore(scenario.concurrency)
    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=False,
        headers=headers,
        transport=transport,
    ) as client:
        selected_model = model or await _discover_model(client, base_url)

        async def send(index: int) -> ScenarioRequestResult:
            if index and launch_interval_seconds:
                await _async_sleep(index * launch_interval_seconds)
            async with semaphore:
                started = perf_counter()
                prompt_tokens = None
                completion_tokens = None
                try:
                    response = await client.post(
                        f"{base_url}/chat/completions",
                        json={
                            "model": selected_model,
                            "messages": [{"role": "user", "content": _prompt(scenario, index)}],
                            "max_tokens": scenario.max_tokens,
                            "temperature": 0,
                            "stream": False,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    usage = payload.get("usage") if isinstance(payload, dict) else None
                    usage = usage if isinstance(usage, dict) else {}
                    prompt_tokens = _integer(usage.get("prompt_tokens"))
                    completion_tokens = _integer(usage.get("completion_tokens"))
                    failure = None
                    succeeded = True
                except (httpx.HTTPError, ValueError) as exc:
                    response = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
                    failure = _safe_failure(exc)
                    succeeded = False
                return ScenarioRequestResult(
                    index=index,
                    succeeded=succeeded,
                    status_code=response.status_code if response is not None else None,
                    latency_ms=(perf_counter() - started) * 1000,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    error=failure,
                )

        requests = tuple(
            await asyncio.gather(*(send(index) for index in range(scenario.request_count)))
        )
    return ScenarioRunResult(
        endpoint=base_url,
        model=selected_model,
        scenario=scenario,
        requests=requests,
    )


def _validate_demo_stages(stages: tuple[DemoStage, ...]) -> None:
    if not stages:
        raise ScenarioError("demo must contain at least one stage")
    request_count = 0
    requested_tokens = 0
    for stage in stages:
        _validate_scenario(stage.scenario)
        if not 0 <= stage.launch_interval_seconds <= MAX_LAUNCH_INTERVAL_SECONDS:
            raise ScenarioError("demo stage launch interval is outside the hard bounds")
        request_count += stage.scenario.request_count
        requested_tokens += stage.scenario.request_count * stage.scenario.max_tokens
    if request_count > MAX_DEMO_REQUESTS:
        raise ScenarioError(f"demo exceeds its {MAX_DEMO_REQUESTS}-request cap")
    if requested_tokens > MAX_DEMO_OUTPUT_TOKENS:
        raise ScenarioError(f"demo exceeds its {MAX_DEMO_OUTPUT_TOKENS}-output-token cap")


async def run_demo_transition(
    endpoint: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 120.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DemoRunResult:
    """Run the fixed paced-baseline, queue-pressure, and paced-recovery stages."""

    _validate_demo_stages(DEMO_STAGES)
    selected_model = model
    results = []
    for stage in DEMO_STAGES:
        result = await run_scenario(
            endpoint,
            stage.scenario,
            model=selected_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            launch_interval_seconds=stage.launch_interval_seconds,
            transport=transport,
        )
        selected_model = result.model
        results.append(DemoStageResult(stage=stage, result=result))
    return DemoRunResult(
        endpoint=results[0].result.endpoint,
        model=results[0].result.model,
        stages=tuple(results),
    )
