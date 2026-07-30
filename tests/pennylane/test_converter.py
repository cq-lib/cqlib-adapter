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

from __future__ import annotations

import pennylane as qml
import pytest

pytest.importorskip("cqlib")
from pennylane.tape import QuantumScript

from cqlib_adapter.common import AdapterConversionError, CompilationOptions
from cqlib_adapter.pennylane import (
    X2P,
    compile_pennylane_circuit,
    cqlib_to_pennylane,
    pennylane_to_cqlib,
)

pytestmark = pytest.mark.pennylane

NATIVE_BASIS = (
    "RZ",
    "X2P",
    "X2M",
    "Y2P",
    "Y2M",
    "XY2P",
    "XY2M",
    "CZ",
    "GPHASE",
)


def operation_names(bundle: object) -> list[str]:
    return [
        item.instruction.instruction.name.lower()  # type: ignore[attr-defined]
        for item in bundle.circuit.operations  # type: ignore[attr-defined]
    ]


def test_quantum_script_translates_operations_wires_and_measurements() -> None:
    tape = QuantumScript(
        [
            qml.Hadamard("left"),
            qml.CNOT(("left", "right")),
            qml.RZ(0.25, "right"),
            X2P(wires="left"),
        ],
        [qml.counts(wires=("left", "right"))],
        shots=100,
    )

    bundle = pennylane_to_cqlib(tape, wire_order=("left", "right"))

    assert operation_names(bundle) == ["h", "cx", "rz", "x2p", "measure_bit", "measure_bit"]
    assert bundle.metadata.framework == "pennylane"
    assert bundle.metadata.extras["wire_order"] == ("left", "right")
    assert [
        (item.logical_qubit, item.classical_bit) for item in bundle.metadata.measurements.slots
    ] == [
        ("wire0", 0),
        ("wire1", 1),
    ]


def test_compile_pennylane_circuit_uses_real_cqlib_native_lowering() -> None:
    tape = QuantumScript(
        [qml.Hadamard(0), qml.CNOT((0, 1))],
        [qml.counts(wires=(0, 1))],
        shots=20,
    )

    artifact = compile_pennylane_circuit(
        tape,
        options=CompilationOptions(target_basis=NATIVE_BASIS, seed=13),
    )

    assert "CZ Q0 Q1" in artifact.qcis
    assert artifact.qcis.count("M Q") == 2
    assert [(item.physical_qubit, item.classical_bit) for item in artifact.measurements] == [
        (0, 0),
        (1, 1),
    ]
    assert all(
        item.instruction.instruction.name.upper() in set(NATIVE_BASIS) | {"MEASURE_BIT"}
        for item in artifact.circuit.operations
    )


def test_compile_respects_real_cqlib_device_topology() -> None:
    from cqlib_adapter.common import NormalizedDevice
    from cqlib_adapter.pennylane.testing import MockCloudBackend

    normalized = NormalizedDevice.from_backend(MockCloudBackend([], size=3))
    tape = QuantumScript(
        [qml.Hadamard(0), qml.CNOT((0, 2))],
        [qml.counts(wires=(0, 2))],
        shots=20,
    )

    artifact = compile_pennylane_circuit(tape, wire_order=(0, 1, 2), device=normalized)

    assert any(step.name == "route.sabre" and not step.skipped for step in artifact.steps)
    for operation in artifact.circuit.operations:
        qubits = tuple(qubit.index for qubit in operation.qubits)
        if len(qubits) == 2:
            assert normalized.supports_coupling(*qubits, either_direction=True)


def test_cqlib_roundtrip_restores_pennylane_operations_and_measurement() -> None:
    tape = QuantumScript(
        [qml.PauliX("a"), qml.CZ(("a", "b")), qml.RY(0.4, "b")],
        [qml.probs(wires=("a", "b"))],
        shots=50,
    )
    bundle = pennylane_to_cqlib(tape, wire_order=("a", "b"))

    restored = cqlib_to_pennylane(
        bundle.circuit,
        metadata=bundle.metadata,
        shots=50,
        measurement="probs",
    )

    assert [item.name for item in restored.operations] == ["PauliX", "CZ", "RY"]
    assert tuple(restored.wires) == ("a", "b")
    assert type(restored.measurements[0]).__name__ == "ProbabilityMP"


def test_pauli_observable_measurements_insert_basis_rotations() -> None:
    tape = QuantumScript(
        [],
        [qml.sample(qml.X(0)), qml.counts(qml.Y(1))],
        shots=10,
    )

    bundle = pennylane_to_cqlib(tape, wire_order=(0, 1))

    assert operation_names(bundle) == [
        "h",
        "sdg",
        "h",
        "measure_bit",
        "measure_bit",
    ]
    assert bundle.metadata.extras["measurement_bases"] == ("X", "Y")


def test_incompatible_or_unsupported_observable_measurements_are_rejected() -> None:
    conflicting = QuantumScript(
        [],
        [qml.counts(wires=0), qml.sample(qml.X(0))],
        shots=10,
    )
    unsupported = QuantumScript(
        [],
        [qml.expval(qml.Hadamard(0))],
        shots=10,
    )

    with pytest.raises(AdapterConversionError, match="incompatible"):
        pennylane_to_cqlib(conflicting)
    with pytest.raises(AdapterConversionError, match="PauliX"):
        pennylane_to_cqlib(unsupported)


def test_missing_measurement_and_incomplete_wire_order_are_rejected() -> None:
    with pytest.raises(AdapterConversionError, match="final measurement"):
        pennylane_to_cqlib(QuantumScript([qml.PauliX(0)], [], shots=10))
    tape = QuantumScript([qml.CNOT((0, 1))], [qml.counts(wires=(0, 1))], shots=10)
    with pytest.raises(AdapterConversionError, match="omits"):
        pennylane_to_cqlib(tape, wire_order=(0,))


def test_unsupported_operation_reports_its_tape_index() -> None:
    tape = QuantumScript(
        [qml.QubitUnitary([[1, 0], [0, 1]], wires=0)], [qml.counts(wires=0)], shots=10
    )
    with pytest.raises(AdapterConversionError, match="operation at index 0"):
        pennylane_to_cqlib(tape)
