"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from infertop import __version__
from infertop.collector import CollectionError, collect_endpoint, collect_files
from infertop.prometheus import MetricsParseError
from infertop.report import render_json, render_text
from infertop.rules import diagnose


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infertop")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnose_parser = subparsers.add_parser("diagnose", help="diagnose a live URL or saved scrape")
    diagnose_parser.add_argument("target", help="server base URL, /metrics URL, or metrics file")
    diagnose_parser.add_argument(
        "--previous",
        type=Path,
        help="earlier metrics file (offline counter deltas)",
    )
    diagnose_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="seconds between live scrapes, or between fixture files (default: 1)",
    )
    diagnose_parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP timeout in seconds (default: 5)",
    )
    diagnose_parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.target.startswith(("http://", "https://")):
            if args.previous is not None:
                raise CollectionError("--previous is only valid with a metrics file")
            observation = collect_endpoint(
                args.target,
                interval_seconds=args.interval,
                timeout_seconds=args.timeout,
            )
        else:
            observation = collect_files(
                Path(args.target),
                previous_path=args.previous,
                interval_seconds=args.interval,
            )
        findings = diagnose(observation)
        output = (
            render_json(observation, findings) if args.json else render_text(observation, findings)
        )
        print(output)
        return 0
    except (CollectionError, MetricsParseError, OSError, ValueError) as exc:
        print(f"infertop: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
