"""Optional read-only NVIDIA Management Library collection."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from infertop.schema import GpuDeviceSnapshot


class HardwareCollectionError(RuntimeError):
    """Raised when explicitly requested local hardware telemetry is unavailable."""


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
