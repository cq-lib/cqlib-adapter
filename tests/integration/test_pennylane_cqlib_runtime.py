from __future__ import annotations

from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pennylane as qml
import pytest
from cqlib import Circuit

from cqlib_adapter.pennylane import (
    CqlibSimulatorDevice,
    compile_pennylane_circuit,
    pennylane_to_cqlib,
)

pytestmark = [pytest.mark.integration, pytest.mark.pennylane]


def test_pennylane_converter_returns_rust_backed_cqlib_circuit() -> None:
    import cqlib._native as native

    tape = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.CNOT((0, 1))],
        [qml.counts(wires=(0, 1))],
        shots=20,
    )
    bundle = pennylane_to_cqlib(tape)
    assert isinstance(bundle.circuit, Circuit)
    assert any(str(Path(native.__file__).resolve()).endswith(item) for item in EXTENSION_SUFFIXES)
    assert bundle.circuit.validate() is None


def test_pennylane_compiler_calls_real_cqlib_compile_and_qcis() -> None:
    tape = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.CNOT((0, 1))],
        [qml.counts(wires=(0, 1))],
        shots=20,
    )
    artifact = compile_pennylane_circuit(tape)
    assert artifact.circuit.validate() is None
    assert "M Q0" in artifact.qcis
    assert "M Q1" in artifact.qcis
    assert artifact.steps


def test_pennylane_qnode_011_runs_through_real_cqlib_statevector() -> None:
    device = CqlibSimulatorDevice(wires=3)

    @qml.qnode(device, shots=32)
    def circuit():
        qml.PauliX(1)
        qml.PauliX(2)
        return qml.counts(wires=(0, 1, 2))

    assert circuit() == {"011": 32}
    assert device.simulator_calls
    assert device.last_task_ids == ("local-pennylane-task-1",)
