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

"""Offline Cirq Sampler powered by the real cqlib statevector simulator."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

import cirq
from cqlib import Circuit
from cqlib.circuit import Instruction, StandardGate
from cqlib.device import Device, ExecutionResult
from cqlib.ir.qcis import loads as load_qcis
from cqlib.qis.state import Statevector

from cqlib_adapter.common import (
    AdapterConversionError,
    CompilationArtifact,
    TianyanConnector,
)
from cqlib_adapter.common.typing import PlatformLike

from .converter import cirq_to_cqlib
from .sampler import TianyanSampler


@dataclass(frozen=True, slots=True)
class CirqStatevectorResult:
    """Exact cqlib state reordered to the requested Cirq qubit order."""

    data: tuple[complex, ...]
    qcis: str
    artifact: CompilationArtifact
    physical_qubits: tuple[int, ...]
    qubit_order: tuple[cirq.Qid, ...]
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
        measured.append(int(parts[1][1:]))
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
        timeout_secs: float | None = None,
        poll_interval_secs: float = 5.0,
    ) -> list[ExecutionResult]:
        del timeout_secs, poll_interval_secs
        return [self._result]


class _LocalCqlibBackend:
    def __init__(self, num_qubits: int) -> None:
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        self.name = "local-cirq-cqlib-simulator"
        self.display_name = "Local Cirq cqlib Statevector Simulator"
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
            raise ValueError("local simulator accepts one Cirq circuit per task")
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
        result = ExecutionResult.from_counts(
            f"local-cirq-task-{self._counter}",
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
        self.backend = backend

    def list_backends(self) -> list[_LocalCqlibBackend]:
        return [self.backend]

    def get_backend(self, name: str) -> _LocalCqlibBackend:
        if name != self.backend.name:
            raise KeyError(name)
        return self.backend


class CqlibSimulatorSampler(TianyanSampler):
    """No-network Cirq Sampler using cqlib compilation and Statevector."""

    def __init__(self, num_qubits: int, **kwargs: Any) -> None:
        transport = _LocalCqlibBackend(num_qubits)
        connector = TianyanConnector(cast(PlatformLike, _LocalPlatform(transport)))
        super().__init__(connector, connector.get_device(transport.name), **kwargs)
        self._local_transport = transport

    def run_statevector(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        qubit_order: cirq.QubitOrderOrList | None = None,
    ) -> CirqStatevectorResult:
        """Compile a Cirq circuit and return exact pre-measurement amplitudes."""

        if not isinstance(circuit, cirq.AbstractCircuit):
            raise TypeError("circuit must be a cirq.AbstractCircuit")

        measurement_seen = False
        for operation in circuit.all_operations():
            is_measurement = isinstance(
                operation.gate,
                cirq.MeasurementGate,
            )
            if is_measurement:
                measurement_seen = True
            elif measurement_seen:
                raise AdapterConversionError("run_statevector accepts only terminal measurements")

        all_qubits = circuit.all_qubits()
        order_spec = self._qubit_order if qubit_order is None else qubit_order
        order = tuple(
            cirq.QubitOrder.as_qubit_order(
                cirq.QubitOrder.DEFAULT if order_spec is None else order_spec
            ).order_for(all_qubits)
        )
        num_qubits = self._normalized_device.num_qubits
        if len(order) != num_qubits:
            raise ValueError(
                f"statevector circuit must use every simulator qubit ({len(order)} != {num_qubits})"
            )

        unitary = cirq.drop_terminal_measurements(circuit)
        measured = cirq.Circuit(unitary)
        measured.append(
            cirq.measure(
                *order,
                key="__cqlib_statevector__",
            )
        )
        artifact = self._compiler.compile(
            cirq_to_cqlib(
                measured,
                qubit_order=order,
            ),
            device=self._normalized_device,
            options=self._compilation_options,
            circuit_index=0,
        )
        unitary_qcis, _measured = _split_qcis(artifact.qcis)
        simulated = load_qcis(unitary_qcis) if unitary_qcis else Circuit(num_qubits)
        circuit_physical_order = tuple(qubit.index for qubit in simulated.qubits)
        if len(set(circuit_physical_order)) != len(circuit_physical_order) or any(
            qubit < 0 or qubit >= num_qubits for qubit in circuit_physical_order
        ):
            raise ValueError(
                "compiled QCIS statevector contains an invalid physical qubit order "
                f"{circuit_physical_order}"
            )
        if len(artifact.measurements) != num_qubits:
            raise ValueError("run_statevector requires a complete final measurement layout")
        logical_to_physical = tuple(
            measurement.physical_qubit
            for measurement in sorted(
                artifact.measurements,
                key=lambda measurement: measurement.classical_bit,
            )
        )
        if set(logical_to_physical) != set(range(num_qubits)):
            raise ValueError(
                "compiled QCIS statevector requires a complete logical-to-physical layout, "
                f"got {logical_to_physical}"
            )

        state = Statevector.from_circuit(simulated)
        local_data = tuple(complex(amplitude) for amplitude in state.data)
        physical_data = [0j] * (1 << num_qubits)
        for local_index, amplitude in enumerate(local_data):
            physical_index = 0
            for position, physical_qubit in enumerate(circuit_physical_order):
                physical_index |= ((local_index >> position) & 1) << physical_qubit
            physical_data[physical_index] = amplitude

        framework_data = [0j] * (1 << num_qubits)
        for framework_index in range(1 << num_qubits):
            physical_index = 0
            for logical_position in range(num_qubits):
                bit = (framework_index >> (num_qubits - 1 - logical_position)) & 1
                physical_index |= bit << logical_to_physical[logical_position]
            framework_data[framework_index] = physical_data[physical_index]

        return CirqStatevectorResult(
            data=tuple(framework_data),
            qcis=unitary_qcis,
            artifact=artifact,
            physical_qubits=logical_to_physical,
            qubit_order=order,
            num_qubits=num_qubits,
        )

    @property
    def simulator_calls(self) -> tuple[tuple[str, list[str], int], ...]:
        return tuple(self._local_transport.calls)


__all__ = ["CirqStatevectorResult", "CqlibSimulatorSampler"]
