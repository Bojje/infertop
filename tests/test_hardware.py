from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from infertop.hardware import HardwareCollectionError, collect_nvml_gpus
from infertop.report import render_json, render_text
from infertop.rules import diagnose
from infertop.schema import InferenceObservation, InferenceSnapshot


class FakeNVMLError(Exception):
    pass


@dataclass
class FakeUtilization:
    gpu: int
    memory: int


@dataclass
class FakeMemory:
    used: int
    total: int


class FakeBindings:
    NVMLError = FakeNVMLError

    def __init__(self, *, fail_initialization: bool = False) -> None:
        self.fail_initialization = fail_initialization
        self.initialized = False
        self.shutdown = False

    def nvmlInit(self) -> None:
        if self.fail_initialization:
            raise FakeNVMLError("blocked")
        self.initialized = True

    def nvmlShutdown(self) -> None:
        self.shutdown = True

    def nvmlDeviceGetCount(self) -> int:
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetName(self, _handle: int) -> bytes:
        return b"NVIDIA GeForce RTX 5080"

    def nvmlDeviceGetUUID(self, _handle: int) -> str:
        return "GPU-test"

    def nvmlDeviceGetUtilizationRates(self, _handle: int) -> FakeUtilization:
        return FakeUtilization(gpu=92, memory=81)

    def nvmlDeviceGetMemoryInfo(self, _handle: int) -> FakeMemory:
        return FakeMemory(used=12, total=16)

    def nvmlDeviceGetPowerUsage(self, _handle: int) -> int:
        return 270_000

    def nvmlDeviceGetEnforcedPowerLimit(self, _handle: int) -> int:
        return 360_000


def test_collects_read_only_nvml_telemetry_and_shuts_down() -> None:
    bindings = FakeBindings()

    gpus = collect_nvml_gpus(bindings)

    assert bindings.initialized
    assert bindings.shutdown
    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 5080"
    assert gpus[0].gpu_utilization == pytest.approx(0.92)
    assert gpus[0].memory_io_utilization == pytest.approx(0.81)
    assert gpus[0].vram_usage == pytest.approx(0.75)
    assert gpus[0].power_watts == pytest.approx(270)
    assert gpus[0].power_ratio == pytest.approx(0.75)


def test_translates_nvml_initialization_failure() -> None:
    with pytest.raises(HardwareCollectionError, match="could not initialize NVML"):
        collect_nvml_gpus(FakeBindings(fail_initialization=True))


def test_reports_local_hardware_in_text_and_json() -> None:
    gpu = collect_nvml_gpus(FakeBindings())[0]
    observation = InferenceObservation(
        current=InferenceSnapshot(
            source="fixture",
            captured_at=0,
            engine="vllm",
            gpus=(gpu,),
        )
    )
    findings = diagnose(observation)

    text = render_text(observation, findings)
    payload = json.loads(render_json(observation, findings))

    assert "Hardware: local NVML" in text
    assert "NVIDIA GeForce RTX 5080" in text
    assert payload["hardware"]["source"] == "local_nvml"
    assert payload["hardware"]["gpus"][0]["vram_usage"] == pytest.approx(0.75)
