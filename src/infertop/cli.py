"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from infertop import __version__
from infertop.collector import CollectionError, collect_endpoint, collect_file_series, collect_files
from infertop.probe import ProbeError, probe_endpoint
from infertop.prometheus import MetricsParseError
from infertop.report import render_json, render_probe_json, render_probe_text, render_text
from infertop.rules import Finding, Severity, diagnose

_SEVERITY_RANK = {
    Severity.HEALTHY: 0,
    Severity.INFO: 1,
    Severity.WARNING: 2,
    Severity.CRITICAL: 3,
}


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
        "--intermediate",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="metrics file between --previous and target; repeat in chronological order",
    )
    diagnose_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="seconds between live scrapes, or between fixture files (default: 1)",
    )
    diagnose_parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="number of live scrapes; ignored for fixture files (default: 3)",
    )
    diagnose_parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP timeout in seconds (default: 5)",
    )
    diagnose_parser.add_argument(
        "--nvml",
        action="store_true",
        help="fuse read-only local NVIDIA telemetry (requires infertop[nvml])",
    )
    diagnose_parser.add_argument(
        "--fail-on",
        choices=("info", "warning", "critical"),
        help="exit 1 when a finding is at or above this severity (default: report only)",
    )
    diagnose_parser.add_argument("--json", action="store_true", help="emit JSON")
    probe_parser = subparsers.add_parser(
        "probe",
        help="send one bounded OpenAI-compatible inference request",
    )
    probe_parser.add_argument("target", help="server base URL or /v1 URL")
    probe_parser.add_argument("--model", help="served model id (default: discover via /v1/models)")
    probe_parser.add_argument(
        "--prompt",
        default="Reply with exactly the word OK.",
        help="probe prompt",
    )
    probe_parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="maximum output tokens, capped at 256 (default: 8)",
    )
    probe_parser.add_argument(
        "--api-key-env",
        default="INFERTOP_API_KEY",
        help="environment variable containing the API key (default: INFERTOP_API_KEY)",
    )
    probe_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30)",
    )
    probe_parser.add_argument("--json", action="store_true", help="emit JSON")
    watch_parser = subparsers.add_parser(
        "watch",
        help="continuously render ranked findings (requires infertop[tui])",
    )
    watch_parser.add_argument("target", help="server base URL or /metrics URL")
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between scrapes (default: 2)",
    )
    watch_parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="rolling observation window size (default: 3)",
    )
    watch_parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP timeout in seconds (default: 5)",
    )
    watch_parser.add_argument(
        "--nvml",
        action="store_true",
        help="fuse read-only local NVIDIA telemetry (requires infertop[nvml,tui])",
    )
    return parser


def diagnosis_exit_code(
    findings: Iterable[Finding],
    fail_on: str | None,
) -> int:
    """Return 1 when any finding meets an explicitly selected severity threshold."""

    if fail_on is None:
        return 0
    threshold = _SEVERITY_RANK[Severity(fail_on)]
    return int(any(_SEVERITY_RANK[finding.severity] >= threshold for finding in findings))


def _run_diagnose(args: argparse.Namespace) -> tuple[str, int]:
    if args.target.startswith(("http://", "https://")):
        if args.previous is not None or args.intermediate:
            raise CollectionError(
                "--previous and --intermediate are only valid with a metrics file"
            )
        observation = collect_endpoint(
            args.target,
            interval_seconds=args.interval,
            timeout_seconds=args.timeout,
            sample_count=args.samples,
            include_nvml=args.nvml,
        )
    else:
        if args.nvml:
            raise CollectionError("--nvml is only valid with a live endpoint")
        if args.intermediate:
            if args.previous is None:
                raise CollectionError("--intermediate requires --previous")
            observation = collect_file_series(
                (args.previous, *args.intermediate, Path(args.target)),
                interval_seconds=args.interval,
            )
        else:
            observation = collect_files(
                Path(args.target),
                previous_path=args.previous,
                interval_seconds=args.interval,
            )
    findings = diagnose(observation)
    output = render_json(observation, findings) if args.json else render_text(observation, findings)
    return output, diagnosis_exit_code(findings, args.fail_on)


def _run_probe(args: argparse.Namespace) -> str:
    api_key = os.environ.get(args.api_key_env)
    result = probe_endpoint(
        args.target,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        api_key=api_key,
        timeout_seconds=args.timeout,
    )
    return render_probe_json(result) if args.json else render_probe_text(result)


def _run_watch(args: argparse.Namespace) -> None:
    try:
        from infertop.tui import run_watch
    except ImportError as exc:
        raise RuntimeError('watch requires: pip install "infertop[tui]"') from exc
    run_watch(
        args.target,
        interval_seconds=args.interval,
        timeout_seconds=args.timeout,
        sample_count=args.samples,
        include_nvml=args.nvml,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "watch":
            _run_watch(args)
            return 0
        if args.command == "diagnose":
            output, exit_code = _run_diagnose(args)
        else:
            output = _run_probe(args)
            exit_code = 0
        print(output)
        return exit_code
    except (
        CollectionError,
        MetricsParseError,
        ProbeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"infertop: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
