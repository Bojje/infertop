"""Execute one bounded traffic preset against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from infertop.scenarios import (
    DEMO_SCENARIO_NAME,
    DEMO_STAGES,
    MAX_DEMO_OUTPUT_TOKENS,
    MAX_DEMO_REQUESTS,
    SCENARIOS,
    DemoRunResult,
    ScenarioError,
    configured_scenario,
    run_demo_transition,
    run_scenario,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted((*SCENARIOS, DEMO_SCENARIO_NAME)))
    parser.add_argument("endpoint", help="server base URL or /v1 URL")
    parser.add_argument("--model", help="served model id (default: discover /v1/models)")
    parser.add_argument("--requests", type=int, help="override the preset request count")
    parser.add_argument("--concurrency", type=int, help="override preset concurrency")
    parser.add_argument("--prompt-words", type=int, help="override approximate prompt words")
    parser.add_argument("--max-tokens", type=int, help="override maximum output tokens")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key-env", default="INFERTOP_API_KEY")
    parser.add_argument(
        "--confirm-active-load",
        action="store_true",
        help="required confirmation that this command sends inference requests",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.confirm_active_load:
            raise ScenarioError("refusing active load without --confirm-active-load")
        if args.scenario == DEMO_SCENARIO_NAME:
            if any(
                value is not None
                for value in (args.requests, args.concurrency, args.prompt_words, args.max_tokens)
            ):
                raise ScenarioError("demo-transition stages are fixed and do not accept overrides")
            demo_requests = sum(stage.scenario.request_count for stage in DEMO_STAGES)
            demo_tokens = sum(
                stage.scenario.request_count * stage.scenario.max_tokens for stage in DEMO_STAGES
            )
            print(
                f"Active demo ceiling: {demo_requests}/{MAX_DEMO_REQUESTS} requests; "
                f"{demo_tokens}/{MAX_DEMO_OUTPUT_TOKENS} requested output tokens",
                file=sys.stderr,
            )
            result = asyncio.run(
                run_demo_transition(
                    args.endpoint,
                    model=args.model,
                    api_key=os.environ.get(args.api_key_env),
                    timeout_seconds=args.timeout,
                )
            )
            return _print_demo_result(result, as_json=args.json)
        scenario = configured_scenario(
            args.scenario,
            request_count=args.requests,
            concurrency=args.concurrency,
            prompt_words=args.prompt_words,
            max_tokens=args.max_tokens,
        )
        result = asyncio.run(
            run_scenario(
                args.endpoint,
                scenario,
                model=args.model,
                api_key=os.environ.get(args.api_key_env),
                timeout_seconds=args.timeout,
            )
        )
    except (ScenarioError, OSError, ValueError) as exc:
        print(f"load_scenario: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Scenario: {scenario.name} ({scenario.description})")
        print(f"Model: {result.model}")
        print(
            f"Requests: {scenario.request_count}; concurrency: {scenario.concurrency}; "
            f"succeeded: {result.succeeded}; failed: {result.failed}"
        )
        if result.p50_latency_ms is not None and result.p95_latency_ms is not None:
            print(
                f"Completion latency: p50 {result.p50_latency_ms:.1f}ms; "
                f"p95 {result.p95_latency_ms:.1f}ms"
            )
        if result.prompt_tokens is not None or result.completion_tokens is not None:
            print(
                f"Reported tokens: prompt {result.prompt_tokens or 0}; "
                f"completion {result.completion_tokens or 0}"
            )
    return int(result.failed > 0)


def _print_demo_result(result: DemoRunResult, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("Demo: healthy baseline -> queue pressure -> healthy recovery")
        print(f"Model: {result.model}")
        print(
            f"Hard ceiling: {result.request_count}/{MAX_DEMO_REQUESTS} requests; "
            f"{result.requested_output_token_ceiling}/{MAX_DEMO_OUTPUT_TOKENS} "
            "requested output tokens"
        )
        for index, stage in enumerate(result.stages, 1):
            print(
                f"Stage {index}/{len(result.stages)}: {stage.stage.name}; "
                f"requests {stage.result.scenario.request_count}; "
                f"concurrency {stage.result.scenario.concurrency}; "
                f"succeeded {stage.result.succeeded}; failed {stage.result.failed}"
            )
    return int(result.failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
