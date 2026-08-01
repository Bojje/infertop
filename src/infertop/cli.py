"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from infertop import __version__
from infertop.collector import CollectionError, collect_endpoint, collect_file_series, collect_files
from infertop.probe import (
    MAX_PROBE_REQUESTS,
    MAX_PROBE_TOTAL_OUTPUT_TOKENS,
    ProbeError,
    probe_endpoint,
    probe_endpoint_repeated,
)
from infertop.prometheus import MetricsParseError
from infertop.prometheus_api import collect_prometheus_range, parse_range_time
from infertop.report import render_json, render_probe_json, render_probe_text, render_text
from infertop.rules import Finding, Severity, diagnose

_SEVERITY_RANK = {
    Severity.HEALTHY: 0,
    Severity.INFO: 1,
    Severity.WARNING: 2,
    Severity.CRITICAL: 3,
}


def _gpu_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated integers") from exc
    if len(indices) < 2 or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError(
            "TP topology requires at least two non-negative GPU indices"
        )
    if tuple(sorted(set(indices))) != indices:
        raise argparse.ArgumentTypeError("GPU indices must be unique and sorted")
    return indices


def _prometheus_label(value: str) -> tuple[str, str]:
    name, separator, label_value = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("Prometheus labels must use NAME=VALUE")
    return name, label_value


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
        "--tp-gpus",
        type=_gpu_indices,
        default=(),
        metavar="INDEX,INDEX",
        help="explicit local tensor-parallel GPU indices for read-only topology diagnosis",
    )
    diagnose_parser.add_argument(
        "--api-key-env",
        default="INFERTOP_API_KEY",
        help="environment variable containing a metrics bearer token (default: INFERTOP_API_KEY)",
    )
    diagnose_parser.add_argument(
        "--prometheus",
        action="store_true",
        help="treat target as a Prometheus server and query a historical range",
    )
    diagnose_parser.add_argument(
        "--start",
        type=parse_range_time,
        help="Prometheus range start as an RFC3339 or Unix timestamp",
    )
    diagnose_parser.add_argument(
        "--end",
        type=parse_range_time,
        help="Prometheus range end as an RFC3339 or Unix timestamp",
    )
    diagnose_parser.add_argument(
        "--step",
        type=float,
        default=None,
        help="Prometheus query resolution in seconds (default: 15, maximum 120 samples)",
    )
    diagnose_parser.add_argument(
        "--prometheus-label",
        type=_prometheus_label,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="exact Prometheus series label filter; repeat to select one endpoint",
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
        "--count",
        type=int,
        default=1,
        help=(
            f"sequential request count, capped at {MAX_PROBE_REQUESTS}; "
            f"count x max-tokens is capped at {MAX_PROBE_TOTAL_OUTPUT_TOKENS} (default: 1)"
        ),
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
    watch_parser.add_argument(
        "--tp-gpus",
        type=_gpu_indices,
        default=(),
        metavar="INDEX,INDEX",
        help="explicit local tensor-parallel GPU indices for read-only topology diagnosis",
    )
    watch_parser.add_argument(
        "--api-key-env",
        default="INFERTOP_API_KEY",
        help="environment variable containing a metrics bearer token (default: INFERTOP_API_KEY)",
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
    if args.prometheus:
        if args.start is None or args.end is None:
            raise CollectionError("--prometheus requires --start and --end")
        if args.previous is not None or args.intermediate:
            raise CollectionError(
                "--previous and --intermediate cannot be combined with --prometheus"
            )
        if args.nvml or args.tp_gpus:
            raise CollectionError("local GPU evidence cannot be combined with --prometheus")
        if args.interval != 1.0 or args.samples != 3:
            raise CollectionError("--interval and --samples cannot be combined with --prometheus")
        label_names = [name for name, _value in args.prometheus_label]
        if len(label_names) != len(set(label_names)):
            raise CollectionError("each --prometheus-label name may be specified only once")
        observation = collect_prometheus_range(
            args.target,
            start=args.start,
            end=args.end,
            step_seconds=args.step if args.step is not None else 15.0,
            timeout_seconds=args.timeout,
            labels=dict(args.prometheus_label),
            api_key=os.environ.get(args.api_key_env),
        )
    elif (
        args.start is not None
        or args.end is not None
        or args.step is not None
        or args.prometheus_label
    ):
        raise CollectionError("--start, --end, --step, and --prometheus-label require --prometheus")
    elif args.target.startswith(("http://", "https://")):
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
            api_key=os.environ.get(args.api_key_env),
            tensor_parallel_gpu_indices=args.tp_gpus,
        )
    else:
        if args.nvml:
            raise CollectionError("--nvml is only valid with a live endpoint")
        if args.tp_gpus:
            raise CollectionError("--tp-gpus is only valid with a live endpoint")
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
    options = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "api_key": api_key,
        "timeout_seconds": args.timeout,
    }
    result = (
        probe_endpoint(args.target, **options)
        if args.count == 1
        else probe_endpoint_repeated(args.target, request_count=args.count, **options)
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
        api_key=os.environ.get(args.api_key_env),
        tensor_parallel_gpu_indices=args.tp_gpus,
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
