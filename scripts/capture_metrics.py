"""Capture a chronological raw metrics series and reproducibility manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from infertop.capture import CaptureError, capture_metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", help="server base URL or /metrics URL")
    parser.add_argument(
        "output", type=Path, help="new output directory; existing paths are refused"
    )
    parser.add_argument("--scenario", required=True, help="scenario name recorded in provenance")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--engine", choices=("vllm", "sglang"))
    parser.add_argument("--engine-version")
    parser.add_argument("--model")
    parser.add_argument("--server-command")
    parser.add_argument("--api-key-env", default="INFERTOP_API_KEY")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = capture_metrics(
            args.endpoint,
            args.output,
            scenario=args.scenario,
            sample_count=args.samples,
            interval_seconds=args.interval,
            timeout_seconds=args.timeout,
            api_key=os.environ.get(args.api_key_env),
            expected_engine=args.engine,
            engine_version=args.engine_version,
            model=args.model,
            server_command=args.server_command,
        )
    except (CaptureError, OSError, ValueError) as exc:
        print(f"capture_metrics: error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Captured {manifest.sample_count} {manifest.engine} scrapes to {args.output} "
        f"for scenario {manifest.scenario}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
