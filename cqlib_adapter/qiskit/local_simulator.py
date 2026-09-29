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

"""Offline Qiskit BackendV2 powered by the cqlib statevector simulator.

This module exercises the same Qiskit-to-cqlib compilation, QCIS, Job and
Result path as the Tianyan backend, but replaces cloud transport with local
cqlib simulation. It never authenticates or contacts Tianyan.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from cqlib import Circuit
from cqlib.circuit import Instruction, StandardGate
from cqlib.device import Device, ExecutionResult
from cqlib.ir.qcis import loads as load_qcis
from cqlib.qis.state import Statevector
from qiskit import QuantumCircuit

from cqlib_adapter.common import (
    AdapterConversionError,
    CompilationArtifact,
    CompilationOptions,
    TianyanConnector,
)
from cqlib_adapter.common.typing import PlatformLike

from .backend import TianyanBackend
from .converter import qiskit_to_cqlib


@dataclass(frozen=True, slots=True)
class CqlibStatevectorResult:
    """Exact state produced by native-QCIS simulation in physical-qubit order."""

    data: tuple[complex, ...]
    qcis: str
    artifact: CompilationArtifact
    physical_qubits: tuple[int, ...]
    num_qubits: int


def _native_instructions() -> list[Instruction]:
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


def _split_qcis(qcis: str) -> tuple[str, tuple[int, ...]]:
    unitary_lines: list[str] = []
    measured: list[int] = []
    for raw_line in qcis.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].upper() != "M":
            unitary_lines.append(line)
            continue
        if len(parts) != 2 or not parts[1].upper().startswith("Q"):
            raise ValueError(f"unsupported QCIS measurement directive {raw_line!r}")
        try:
            measured.append(int(parts[1][1:]))
        except ValueError as exc:
            raise ValueError(f"invalid QCIS measurement qubit {parts[1]!r}") from exc
    if not measured:
        raise ValueError("simulator QCIS contains no measurements")
    return "\n".join(unitary_lines), tuple(measured)


class _LocalTaskHandle:
    def __init__(self, result: ExecutionResult, *, device_name: str, shots: int) -> None:
        self.task_ids = [result.task_id]
        self.device_name = device_name
        self.shots = shots
        self._result = result

    def status(self) -> list[ExecutionResult]:
        return [self._result]

    def wait(
        self,
        timeout: float | None = None,
        poll_interval: float = 5.0,
    ) -> list[ExecutionResult]:
        del timeout, poll_interval
        return [self._result]


class _LocalCqlibBackend:
    def __init__(self, num_qubits: int) -> None:
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        self.name = "local-cqlib-simulator"
        self.display_name = "Local cqlib Statevector Simulator"
        self.status = "running"
        self.toll = "free"
        self._device = Device.line(self.name, num_qubits)
        self._device.native_gates = _native_instructions()
        self._counter = 0
        self.calls: list[tuple[str, list[str], int]] = []

    def is_available(self) -> bool:
        return True

    def num_qubits(self) -> int:
        return len(self._device.qubits)

    def device_config(self) -> Device:
        return self._device

    def _submit(self, mode: str, circuits: list[str], shots: int) -> _LocalTaskHandle:
        self.calls.append((mode, list(circuits), shots))
        if len(circuits) != 1:
            raise ValueError("local simulator accepts one metadata-bound circuit per task")

        unitary_qcis, _measured = _split_qcis(circuits[0])
        simulated_circuit = load_qcis(unitary_qcis) if unitary_qcis else Circuit(self.num_qubits())
        physical_order = tuple(qubit.index for qubit in simulated_circuit.qubits)
        if any(qubit < 0 or qubit >= self.num_qubits() for qubit in physical_order):
            raise ValueError("QCIS references a qubit outside the simulator")
        state = Statevector.from_circuit(simulated_circuit)
        samples = state.sample_shots(shots)

        def physical_outcome(sample: Any) -> str:
            local = sample.to_bitstring(state.num_qubits)
            physical = ["0"] * self.num_qubits()
            for position, qubit in enumerate(physical_order):
                physical[qubit] = local[-1 - position]
            return "".join(reversed(physical))

        counts = dict(Counter(physical_outcome(sample) for sample in samples))
        self._counter += 1
        task_id = f"local-cqlib-task-{self._counter}"
        result = ExecutionResult.from_counts(
            task_id,
            list(range(self.num_qubits())),
            shots,
            self.num_qubits(),
            counts,
            backend=self.name,
        )
        return _LocalTaskHandle(result, device_name=self.name, shots=shots)

    def run(self, circuits: list[str], shots: int = 1024) -> _LocalTaskHandle:
        return self._submit("auto", circuits, shots)

    def run_raw(self, circuits: list[str], shots: int = 1024) -> _LocalTaskHandle:
        return self._submit("disabled", circuits, shots)

    def run_with_mode(
        self,
        circuits: list[str],
        shots: int = 1024,
        mode: str = "auto",
    ) -> _LocalTaskHandle:
        return self._submit(mode, circuits, shots)


class _LocalPlatform:
    def __init__(self, backend: _LocalCqlibBackend) -> None:
        self._backend = backend

    def list_backends(self) -> list[_LocalCqlibBackend]:
        return [self._backend]

    def get_backend(self, name: str) -> _LocalCqlibBackend:
        if name != self._backend.name:
            raise KeyError(name)
        return self._backend


class CqlibSimulatorBackend(TianyanBackend):
    """Qiskit BackendV2 using adapter compilation and local cqlib simulation.

    The backend is intended for offline adapter verification. It compiles to
    Tianyan-compatible native QCIS, simulates that compiled circuit locally,
    and returns the same TianyanJob/Qiskit Result types as the cloud backend.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        max_circuits: int | None = 50,
        **kwargs: Any,
    ) -> None:
        transport = _LocalCqlibBackend(num_qubits)
        platform = cast(PlatformLike, _LocalPlatform(transport))
        connector = TianyanConnector(platform)
        super().__init__(
            connector,
            connector.get_device(transport.name),
            max_circuits=max_circuits,
            **kwargs,
        )
        self._local_transport = transport

    def run_statevector(
        self,
        circuit: QuantumCircuit,
        *,
        seed: int | None = None,
    ) -> CqlibStatevectorResult:
        """Compile one unitary Qiskit circuit and return exact cqlib amplitudes.

        Final measurements are removed because a statevector represents the
        pre-measurement pure state. Mid-circuit measurements and reset are
        rejected because they do not define one deterministic pure state.
        Returned amplitude indices use backend physical-qubit indices.
        """

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("circuit must be a qiskit.QuantumCircuit")
        if circuit.num_qubits != self.num_qubits:
            raise ValueError(
                "statevector circuit width must equal backend.num_qubits "
                f"({circuit.num_qubits} != {self.num_qubits})"
            )

        unitary = circuit.remove_final_measurements(inplace=False)
        nonunitary = [
            item.operation.name
            for item in unitary.data
            if item.operation.name in {"measure", "reset"}
        ]
        if nonunitary:
            raise AdapterConversionError(
                "run_statevector does not support mid-circuit measurement or reset"
            )

        artifact = self._compiler.compile(
            qiskit_to_cqlib(unitary),
            device=self._device,
            options=CompilationOptions(seed=seed),
            circuit_index=0,
        )
        simulated_circuit = artifact.circuit
        physical_order = tuple(qubit.index for qubit in simulated_circuit.qubits)
        if any(qubit < 0 or qubit >= self.num_qubits for qubit in physical_order):
            raise ValueError("compiled QCIS references a qubit outside the simulator")

        state = Statevector.from_circuit(simulated_circuit)
        local_data = tuple(complex(amplitude) for amplitude in state.data)
        physical_data = [0j] * (1 << self.num_qubits)
        for local_index, amplitude in enumerate(local_data):
            physical_index = 0
            for position, physical_qubit in enumerate(physical_order):
                physical_index |= ((local_index >> position) & 1) << physical_qubit
            physical_data[physical_index] = amplitude

        return CqlibStatevectorResult(
            data=tuple(physical_data),
            qcis=artifact.qcis,
            artifact=artifact,
            physical_qubits=physical_order,
            num_qubits=self.num_qubits,
        )

    @property
    def simulator_calls(self) -> tuple[tuple[str, list[str], int], ...]:
        """QCIS submissions observed by the local simulator transport."""

        return tuple(self._local_transport.calls)


__all__ = ["CqlibSimulatorBackend", "CqlibStatevectorResult"]
