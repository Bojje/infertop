"""Execute one bounded traffic preset against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from infertop.scenarios import SCENARIOS, ScenarioError, configured_scenario, run_scenario


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
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
        scenario = configured_scenario(
            args.scenario,
            request_count=args.requests,
            concurrency=args.concurrency,
            prompt_words=args.prompt_words,
            max_tokens=args.max_tokens,
        )
        if not args.confirm_active_load:
            raise ScenarioError("refusing active load without --confirm-active-load")
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


if __name__ == "__main__":
    raise SystemExit(main())
