"""Offline Tianyan doubles that still use real cqlib Device/ExecutionResult."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cqlib.circuit import Instruction, StandardGate
from cqlib.device import Device, ExecutionResult

from cqlib_adapter.common import TianyanConnector
from cqlib_adapter.common.typing import PlatformLike

from .backend import TianyanBackend


def native_instructions() -> list[Instruction]:
    return [
        Instruction.from_standard_gate(getattr(StandardGate, name))
        for name in (
            "RZ",
            "X2P",
            "X2M",
            "Y2P",
            "Y2M",
            "XY2P",
            "XY2M",
            "CZ",
            "GPhase",
        )
    ]


@dataclass(frozen=True)
class ResultSpec:
    counts: dict[str, int]
    qubits: tuple[int, ...]
    status_ready: bool = False


class MockTaskHandle:
    def __init__(
        self,
        result: ExecutionResult,
        *,
        device_name: str,
        shots: int,
        status_ready: bool,
    ) -> None:
        self.task_ids = [result.task_id]
        self.device_name = device_name
        self.shots = shots
        self._result = result
        self._status_ready = status_ready
        self.wait_calls: list[tuple[float | None, float]] = []

    def status(self) -> list[ExecutionResult]:
        return [self._result] if self._status_ready else []

    def wait(
        self,
        timeout_secs: float | None = None,
        poll_interval_secs: float = 5.0,
    ) -> list[ExecutionResult]:
        self.wait_calls.append((timeout_secs, poll_interval_secs))
        return [self._result]


class MockCloudBackend:
    def __init__(
        self,
        specs: list[ResultSpec],
        *,
        name: str = "mock-qpu",
        size: int = 3,
        available: bool = True,
    ) -> None:
        self.name = name
        self.display_name = "Mock Tianyan QPU"
        self.status = "running" if available else "offline"
        self.toll = "free"
        self._available = available
        self._config = Device.line(name, size)
        self._config.native_gates = native_instructions()
        self._specs = list(specs)
        self.calls: list[tuple[str, list[str], int]] = []
        self.handles: list[MockTaskHandle] = []
        self._counter = 0

    def is_available(self) -> bool:
        return self._available

    def num_qubits(self) -> int:
        return len(self._config.qubits)

    def device_config(self) -> Device:
        return self._config

    def _submit(self, mode: str, circuits: list[str], shots: int) -> MockTaskHandle:
        self.calls.append((mode, list(circuits), shots))
        if len(circuits) != 1:
            raise AssertionError("the adapter submits one metadata-bound circuit per handle")
        if not self._specs:
            raise AssertionError("no mock result was configured")
        spec = self._specs.pop(0)
        if sum(spec.counts.values()) != shots:
            raise AssertionError("mock counts must sum to submitted shots")
        self._counter += 1
        task_id = f"mock-task-{self._counter}"
        result = ExecutionResult.from_counts(
            task_id,
            list(spec.qubits),
            shots,
            self.num_qubits(),
            spec.counts,
        )
        handle = MockTaskHandle(
            result,
            device_name=self.name,
            shots=shots,
            status_ready=spec.status_ready,
        )
        self.handles.append(handle)
        return handle

    def run(self, circuits: list[str], shots: int = 1024) -> MockTaskHandle:
        return self._submit("auto", circuits, shots)

    def run_raw(self, circuits: list[str], shots: int = 1024) -> MockTaskHandle:
        return self._submit("disabled", circuits, shots)

    def run_with_mode(
        self,
        circuits: list[str],
        shots: int = 1024,
        mode: str = "auto",
    ) -> MockTaskHandle:
        return self._submit(mode, circuits, shots)


class MockPlatform:
    def __init__(self, backend: MockCloudBackend) -> None:
        self.backend = backend

    def list_backends(self) -> list[MockCloudBackend]:
        return [self.backend]

    def get_backend(self, name: str) -> MockCloudBackend:
        if name != self.backend.name:
            raise KeyError(name)
        return self.backend


def make_qiskit_backend(
    specs: list[ResultSpec],
    *,
    size: int = 3,
    available: bool = True,
    max_circuits: int | None = 50,
) -> tuple[TianyanBackend, MockCloudBackend]:
    cloud_backend = MockCloudBackend(specs, size=size, available=available)
    connector = TianyanConnector(cast(PlatformLike, MockPlatform(cloud_backend)))
    backend = TianyanBackend.from_connector(
        connector,
        cloud_backend.name,
        max_circuits=max_circuits,
    )
    return backend, cloud_backend
