"""Optional read-only NVIDIA Management Library collection."""

from __future__ import annotations

import re
import subprocess
from contextlib import suppress
from typing import Any, Protocol

from infertop.schema import GpuDeviceSnapshot, GpuTopology, GpuTopologyLink


class HardwareCollectionError(RuntimeError):
    """Raised when explicitly requested local hardware telemetry is unavailable."""


class TopologyParseError(ValueError):
    """Raised when `nvidia-smi topo -m` output is incomplete or inconsistent."""


class CommandRunner(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]: ...


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_GPU_NAME = re.compile(r"GPU(?P<index>\d+)$")
_LINK_KIND = re.compile(r"(?:X|SYS|NODE|PHB|PXB|PIX|NV\d+|N/A)$")


def _text(value: str | bytes) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _optional(bindings: Any, function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except bindings.NVMLError:
        return None


def collect_nvml_gpus(bindings: Any | None = None) -> tuple[GpuDeviceSnapshot, ...]:
    """Collect one read-only snapshot from every local NVIDIA GPU."""

    if bindings is None:
        try:
            import pynvml as bindings
        except ImportError as exc:
            raise HardwareCollectionError(
                'NVML collection requires: pip install "infertop[nvml]"'
            ) from exc

    try:
        bindings.nvmlInit()
    except bindings.NVMLError as exc:
        raise HardwareCollectionError(f"could not initialize NVML: {exc}") from exc

    try:
        devices = []
        for index in range(bindings.nvmlDeviceGetCount()):
            handle = bindings.nvmlDeviceGetHandleByIndex(index)
            utilization = _optional(bindings, bindings.nvmlDeviceGetUtilizationRates, handle)
            memory = _optional(bindings, bindings.nvmlDeviceGetMemoryInfo, handle)
            power_milliwatts = _optional(bindings, bindings.nvmlDeviceGetPowerUsage, handle)
            power_limit_milliwatts = _optional(
                bindings,
                bindings.nvmlDeviceGetEnforcedPowerLimit,
                handle,
            )
            devices.append(
                GpuDeviceSnapshot(
                    index=index,
                    name=_text(bindings.nvmlDeviceGetName(handle)),
                    uuid=_text(bindings.nvmlDeviceGetUUID(handle)),
                    gpu_utilization=(utilization.gpu / 100.0 if utilization is not None else None),
                    memory_io_utilization=(
                        utilization.memory / 100.0 if utilization is not None else None
                    ),
                    memory_used_bytes=memory.used if memory is not None else None,
                    memory_total_bytes=memory.total if memory is not None else None,
                    power_watts=(
                        power_milliwatts / 1000.0 if power_milliwatts is not None else None
                    ),
                    power_limit_watts=(
                        power_limit_milliwatts / 1000.0
                        if power_limit_milliwatts is not None
                        else None
                    ),
                )
            )
    except bindings.NVMLError as exc:
        raise HardwareCollectionError(f"could not read NVML telemetry: {exc}") from exc
    finally:
        with suppress(bindings.NVMLError):
            bindings.nvmlShutdown()

    if not devices:
        raise HardwareCollectionError("NVML found no local NVIDIA GPUs")
    return tuple(devices)


def parse_nvidia_topology(text: str) -> GpuTopology:
    """Purely parse the GPU matrix from `nvidia-smi topo -m` output."""

    lines = [_ANSI_ESCAPE.sub("", line).strip() for line in text.splitlines()]
    header: tuple[int, ...] | None = None
    header_position = -1
    for position, line in enumerate(lines):
        tokens = line.split()
        gpu_tokens = []
        for token in tokens:
            match = _GPU_NAME.fullmatch(token)
            if match is None:
                if gpu_tokens:
                    break
                continue
            gpu_tokens.append(int(match.group("index")))
        if gpu_tokens:
            header = tuple(gpu_tokens)
            header_position = position
            break
    if header is None:
        raise TopologyParseError("topology table has no GPU header")
    if tuple(sorted(set(header))) != header:
        raise TopologyParseError("topology GPU header must contain unique sorted indices")

    rows: dict[int, tuple[str, ...]] = {}
    for line in lines[header_position + 1 :]:
        tokens = line.split()
        if not tokens:
            continue
        source_match = _GPU_NAME.fullmatch(tokens[0])
        if source_match is None:
            if rows:
                break
            continue
        source = int(source_match.group("index"))
        cells = tuple(tokens[1 : 1 + len(header)])
        if source not in header:
            raise TopologyParseError(f"topology row GPU{source} is absent from the header")
        if source in rows:
            raise TopologyParseError(f"topology contains duplicate GPU{source} rows")
        if len(cells) != len(header) or any(_LINK_KIND.fullmatch(cell) is None for cell in cells):
            raise TopologyParseError(f"topology row GPU{source} has malformed link cells")
        rows[source] = cells
    missing = tuple(index for index in header if index not in rows)
    if missing:
        names = ", ".join(f"GPU{index}" for index in missing)
        raise TopologyParseError(f"topology table is missing rows: {names}")

    for position, index in enumerate(header):
        if rows[index][position] != "X":
            raise TopologyParseError(f"topology diagonal for GPU{index} must be X")

    links = []
    for first_position, first_gpu in enumerate(header):
        for second_position in range(first_position + 1, len(header)):
            second_gpu = header[second_position]
            forward = rows[first_gpu][second_position]
            reverse = rows[second_gpu][first_position]
            if forward != reverse:
                raise TopologyParseError(
                    f"topology link GPU{first_gpu}<->GPU{second_gpu} is asymmetric: "
                    f"{forward}/{reverse}"
                )
            links.append(
                GpuTopologyLink(
                    first_gpu=first_gpu,
                    second_gpu=second_gpu,
                    kind=forward,
                )
            )
    return GpuTopology(gpu_indices=header, links=tuple(links))


def collect_nvidia_topology(runner: CommandRunner = subprocess.run) -> GpuTopology:
    """Run one read-only local NVIDIA topology query and parse its matrix."""

    try:
        result = runner(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HardwareCollectionError(
            f"could not run nvidia-smi topology query: {type(exc).__name__}"
        ) from exc
    if result.returncode != 0:
        raise HardwareCollectionError(
            f"nvidia-smi topology query exited with status {result.returncode}"
        )
    try:
        return parse_nvidia_topology(result.stdout)
    except TopologyParseError as exc:
        raise HardwareCollectionError(f"could not parse NVIDIA topology: {exc}") from exc
