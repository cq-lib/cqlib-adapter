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

"""Small offline doubles for cqlib and cqlib-tianyan contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FakeQubit:
    index: int


@dataclass(frozen=True)
class FakeInstruction:
    name: str


@dataclass(frozen=True)
class FakeValueInstruction:
    instruction: FakeInstruction | None


@dataclass(frozen=True)
class FakeOperation:
    name: str
    indices: tuple[int, ...]

    @property
    def instruction(self) -> FakeValueInstruction:
        return FakeValueInstruction(FakeInstruction(self.name))

    @property
    def qubits(self) -> list[FakeQubit]:
        return [FakeQubit(index) for index in self.indices]


@dataclass
class FakeCircuit:
    operation_specs: tuple[tuple[str, tuple[int, ...]], ...]
    fail_validation: bool = False

    @property
    def operations(self) -> list[FakeOperation]:
        return [FakeOperation(name, indices) for name, indices in self.operation_specs]

    @property
    def qubits(self) -> list[FakeQubit]:
        indices = sorted({index for _, qubits in self.operation_specs for index in qubits})
        return [FakeQubit(index) for index in indices]

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    def validate(self) -> None:
        if self.fail_validation:
            raise ValueError("invalid fake circuit")


@dataclass
class FakeCompileResult:
    circuit: FakeCircuit
    changed: bool = True
    steps: list[str] = field(default_factory=lambda: ["decompose", "route"])


class FakeRuntime:
    def __init__(self, compiled: FakeCircuit | None = None) -> None:
        self.compiled = compiled
        self.calls: list[dict[str, object]] = []
        self.loaded: list[str] = []

    def normal_mode(self) -> str:
        return "normal"

    def enhanced_mode(self) -> str:
        return "enhanced"

    def compile(self, circuit: FakeCircuit, **kwargs: object) -> FakeCompileResult:
        self.calls.append(dict(kwargs))
        return FakeCompileResult(self.compiled or circuit)

    def dumps(self, circuit: FakeCircuit) -> str:
        lines: list[str] = []
        for name, indices in circuit.operation_specs:
            normalized = name.strip().upper()
            if normalized in {"MEASURE", "MEASURE_BIT", "MEASURE_BITS"}:
                lines.extend(f"M Q{index}" for index in indices)
            else:
                lines.append(f"{normalized} {' '.join(f'Q{index}' for index in indices)}".rstrip())
        return "\n".join(lines)

    def loads(self, qcis: str) -> FakeCircuit:
        self.loaded.append(qcis)
        return self.compiled or FakeCircuit(())


@dataclass
class FakeTopology:
    edges: tuple[tuple[int, int], ...]
    size: int

    @property
    def qubits(self) -> list[FakeQubit]:
        return [FakeQubit(index) for index in range(self.size)]

    def successors(self, qubit: FakeQubit) -> list[FakeQubit]:
        return [FakeQubit(target) for source, target in self.edges if source == qubit.index]


@dataclass
class FakeDeviceConfig:
    size: int = 3
    gates: tuple[str, ...] = ("X90", "CZ")
    edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 2), (2, 1))
    invalid: tuple[int, ...] = ()
    name: str = "fake"

    @property
    def qubits(self) -> list[FakeQubit]:
        return [FakeQubit(index) for index in range(self.size)]

    @property
    def invalid_qubits(self) -> list[FakeQubit]:
        return [FakeQubit(index) for index in self.invalid]

    @property
    def native_gates(self) -> list[FakeInstruction]:
        return [FakeInstruction(name) for name in self.gates]

    @property
    def topology(self) -> FakeTopology:
        return FakeTopology(self.edges, self.size)


@dataclass
class FakeStatus:
    kind: str = "completed"
    error_msg: str | None = None
    error_code: int | None = None


@dataclass
class FakeExecutionResult:
    task_id: str
    shots: int
    qubit_indices: tuple[int, ...]
    counts: dict[str, int]
    probabilities: dict[str, float] = field(default_factory=dict)
    status: FakeStatus = field(default_factory=FakeStatus)
    num_qubits: int = 3

    @property
    def qubits(self) -> list[FakeQubit]:
        return [FakeQubit(index) for index in self.qubit_indices]


@dataclass
class FakeHandle:
    task_ids: list[str]
    wait_results: list[FakeExecutionResult]
    status_results: list[FakeExecutionResult] = field(default_factory=list)
    device_name: str = "fake"
    shots: int = 100
    wait_error: Exception | None = None
    wait_calls: list[tuple[float | None, float]] = field(default_factory=list)

    def status(self) -> list[FakeExecutionResult]:
        return list(self.status_results)

    def wait(
        self,
        timeout_secs: float | None = None,
        poll_interval_secs: float = 5.0,
    ) -> list[FakeExecutionResult]:
        self.wait_calls.append((timeout_secs, poll_interval_secs))
        if self.wait_error is not None:
            raise self.wait_error
        return list(self.wait_results)


class FakeBackend:
    def __init__(
        self,
        *,
        name: str = "fake",
        available: bool = True,
        config: FakeDeviceConfig | None = None,
        handles: list[FakeHandle] | None = None,
    ) -> None:
        self.name = name
        self.display_name = f"{name} display"
        self.status = "running" if available else "offline"
        self.toll = "free"
        self._available = available
        self._config = config or FakeDeviceConfig(name=name)
        self._handles = list(handles or [])
        self.calls: list[tuple[str, list[str], int]] = []
        self._counter = 0

    def is_available(self) -> bool:
        return self._available

    def num_qubits(self) -> int:
        return self._config.size

    def device_config(self) -> FakeDeviceConfig:
        return self._config

    def _handle(self, shots: int) -> FakeHandle:
        if self._handles:
            handle = self._handles.pop(0)
            handle.shots = shots
            return handle
        self._counter += 1
        return FakeHandle([f"task-{self._counter}"], [], device_name=self.name, shots=shots)

    def run(self, circuits: list[str], shots: int = 1024) -> FakeHandle:
        self.calls.append(("auto", list(circuits), shots))
        return self._handle(shots)

    def run_raw(self, circuits: list[str], shots: int = 1024) -> FakeHandle:
        self.calls.append(("disabled", list(circuits), shots))
        return self._handle(shots)

    def run_with_mode(
        self,
        circuits: list[str],
        shots: int = 1024,
        mode: str = "auto",
    ) -> FakeHandle:
        self.calls.append((mode, list(circuits), shots))
        return self._handle(shots)


class FakePlatform:
    def __init__(self, backends: list[FakeBackend]) -> None:
        self.backends = {backend.name: backend for backend in backends}

    def list_backends(self) -> list[FakeBackend]:
        return list(self.backends.values())

    def get_backend(self, name: str) -> FakeBackend:
        return self.backends[name]
