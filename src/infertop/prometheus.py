"""Small dependency-free Prometheus text-format parser."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]Inf|NaN)"
    r"(?:\s+\d+)?\s*$"
)
_LABEL_RE = re.compile(r'(?:^|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def label(self, name: str) -> str | None:
        return dict(self.labels).get(name)


class MetricsParseError(ValueError):
    """Raised when exposition text contains a malformed sample."""


def _unescape_label(value: str) -> str:
    return value.replace(r"\n", "\n").replace(r"\\", "\\").replace(r"\"", '"')


def _parse_labels(raw: str | None, line_number: int) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()
    matches = list(_LABEL_RE.finditer(raw))
    if not matches:
        raise MetricsParseError(f"line {line_number}: malformed label set")
    consumed = "".join(match.group(0) for match in matches)
    if re.sub(r"\s+", "", consumed) != re.sub(r"\s+", "", raw):
        raise MetricsParseError(f"line {line_number}: malformed label set")
    return tuple((match.group(1), _unescape_label(match.group(2))) for match in matches)


def parse_metrics(text: str) -> tuple[Sample, ...]:
    """Parse Prometheus/OpenMetrics samples and ignore metadata/comments."""

    samples: list[Sample] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            raise MetricsParseError(f"line {line_number}: malformed metric sample")
        samples.append(
            Sample(
                name=match.group("name"),
                labels=_parse_labels(match.group("labels"), line_number),
                value=float(match.group("value")),
            )
        )
    return tuple(samples)
