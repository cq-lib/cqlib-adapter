# This code is part of cqlib.
#
# Copyright (C) 2025-2026 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Offline Cirq Tianyan doubles using real cqlib device/result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from cqlib.circuit import Instruction, StandardGate
from cqlib.device import Device, ExecutionResult

from cqlib_adapter.common import TianyanConnector
from cqlib_adapter.common.typing import PlatformLike

from .sampler import TianyanSampler


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
        timeout: float | None = None,
        poll_interval: float = 5.0,
    ) -> list[ExecutionResult]:
        self.wait_calls.append((timeout, poll_interval))
        return [self._result]


class MockCloudBackend:
    def __init__(
        self,
        specs: list[ResultSpec],
        *,
        name: str = "mock-cirq-qpu",
        size: int = 3,
        available: bool = True,
    ) -> None:
        self.name = name
        self.display_name = "Mock Cirq Tianyan QPU"
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
            raise AssertionError("adapter submits one Cirq parameter resolution per handle")
        if not self._specs:
            raise AssertionError("no mock Cirq result was configured")
        spec = self._specs.pop(0)
        if sum(spec.counts.values()) != shots:
            raise AssertionError("mock counts must sum to repetitions")
        self._counter += 1
        result = ExecutionResult.from_counts(
            f"mock-cirq-task-{self._counter}",
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


def make_cirq_sampler(
    specs: list[ResultSpec],
    *,
    size: int = 3,
    available: bool = True,
    **sampler_options: Any,
) -> tuple[TianyanSampler, MockCloudBackend]:
    backend = MockCloudBackend(specs, size=size, available=available)
    connector = TianyanConnector(cast(PlatformLike, MockPlatform(backend)))
    sampler = TianyanSampler.from_connector(
        connector,
        backend.name,
        **sampler_options,
    )
    return sampler, backend


__all__ = [
    "MockCloudBackend",
    "MockTaskHandle",
    "ResultSpec",
    "make_cirq_sampler",
    "native_instructions",
]
