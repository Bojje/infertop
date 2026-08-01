from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from infertop.hardware import (
    HardwareCollectionError,
    TopologyParseError,
    collect_nvidia_topology,
    collect_nvml_gpus,
    parse_nvidia_topology,
)
from infertop.report import render_json, render_text
from infertop.rules import diagnose
from infertop.schema import InferenceObservation, InferenceSnapshot

TOPOLOGY_FIXTURES = Path(__file__).parent / "fixtures" / "topology"


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


@pytest.mark.parametrize(
    ("fixture", "indices", "links"),
    (
        ("one_gpu.txt", (0,), ()),
        ("nvlink.txt", (0, 1), ((0, 1, "NV4"),)),
        (
            "cross_numa.txt",
            (0, 1, 2),
            ((0, 1, "PIX"), (0, 2, "SYS"), (1, 2, "NODE")),
        ),
    ),
)
def test_parses_nvidia_topology_fixtures(
    fixture: str,
    indices: tuple[int, ...],
    links: tuple[tuple[int, int, str], ...],
) -> None:
    topology = parse_nvidia_topology((TOPOLOGY_FIXTURES / fixture).read_text())

    assert topology.gpu_indices == indices
    assert tuple((link.first_gpu, link.second_gpu, link.kind) for link in topology.links) == links
    for first_gpu, second_gpu, kind in links:
        assert topology.link_between(second_gpu, first_gpu).kind == kind


def test_topology_parser_strips_terminal_ansi_sequences() -> None:
    topology = parse_nvidia_topology("\x1b[4mGPU0 CPU Affinity\x1b[0m\nGPU0 X N/A\n")

    assert topology.gpu_indices == (0,)


@pytest.mark.parametrize(
    ("text", "message"),
    (
        ("not a topology", "no GPU header"),
        ("GPU0 GPU1\nGPU0 X SYS\n", "missing rows"),
        ("GPU0 GPU1\nGPU0 X SYS\nGPU1 PIX X\n", "asymmetric"),
        ("GPU0\nGPU0 SYS\n", "diagonal"),
    ),
)
def test_topology_parser_rejects_incomplete_or_inconsistent_matrices(
    text: str,
    message: str,
) -> None:
    with pytest.raises(TopologyParseError, match=message):
        parse_nvidia_topology(text)


def test_collects_topology_with_read_only_nvidia_smi_command() -> None:
    calls: list[tuple[object, ...]] = []

    def runner(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(TOPOLOGY_FIXTURES / "nvlink.txt").read_text(),
            stderr="",
        )

    topology = collect_nvidia_topology(runner)

    assert topology.link_between(0, 1).kind == "NV4"
    assert calls == [
        (
            (["nvidia-smi", "topo", "-m"],),
            {
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": 5,
            },
        )
    ]


def test_reports_declared_tp_topology_in_text_and_json() -> None:
    topology = parse_nvidia_topology((TOPOLOGY_FIXTURES / "cross_numa.txt").read_text())
    observation = InferenceObservation(
        current=InferenceSnapshot(source="fixture", captured_at=0, engine="vllm"),
        topology=topology,
        tensor_parallel_gpu_indices=(0, 2),
    )
    findings = diagnose(observation)

    text = render_text(observation, findings)
    payload = json.loads(render_json(observation, findings))

    assert "Topology: local nvidia-smi (declared TP GPUs: GPU0, GPU2)" in text
    assert "GPU 0 <-> GPU 2: SYS" in text
    assert payload["hardware"]["source"] == "local_nvidia_smi"
    assert payload["hardware"]["topology"]["tensor_parallel_gpu_indices"] == [0, 2]
    assert payload["findings"][0]["rule_id"] == "R7_SLOW_TP_TOPOLOGY"
