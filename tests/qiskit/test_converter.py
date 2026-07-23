from __future__ import annotations

from math import pi

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("cqlib")
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Parameter

from cqlib_adapter.common import AdapterConversionError, CompilationOptions
from cqlib_adapter.qiskit import (
    FSimGate,
    RXYGate,
    X2PGate,
    XY2MGate,
    XY2PGate,
    XYGate,
    compile_qiskit_circuit,
    cqlib_to_qiskit,
    qiskit_to_cqlib,
)

pytestmark = pytest.mark.qiskit

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


def operation_names(circuit: object) -> list[str]:
    return [
        operation.instruction.instruction.name  # type: ignore[attr-defined]
        for operation in circuit.operations  # type: ignore[attr-defined]
    ]


def test_qiskit_to_real_cqlib_preserves_gates_parameters_and_measurements() -> None:
    qreg = QuantumRegister(2, "q")
    left = ClassicalRegister(1, "left")
    right = ClassicalRegister(1, "right")
    circuit = QuantumCircuit(qreg, left, right, name="native-mix")
    circuit.append(X2PGate(), [qreg[0]])
    circuit.rz(pi / 3, qreg[0])
    circuit.append(RXYGate(pi / 5, pi / 7), [qreg[1]])
    circuit.cz(qreg[0], qreg[1])
    circuit.measure(qreg[0], right[0])
    circuit.measure(qreg[1], left[0])

    bundle = qiskit_to_cqlib(circuit)

    assert operation_names(bundle.circuit) == [
        "X2P",
        "RZ",
        "RXY",
        "CZ",
        "measure_bit",
        "measure_bit",
    ]
    assert bundle.metadata.circuit_name == "native-mix"
    assert bundle.metadata.measurements.register_sizes == {"left": 1, "right": 1}
    assert [slot.classical_bit for slot in bundle.metadata.measurements.slots] == [1, 0]
    assert [slot.key for slot in bundle.metadata.measurements.slots] == ["right", "left"]


def test_bell_circuit_compiles_through_real_cqlib_to_qcis() -> None:
    circuit = QuantumCircuit(2, 2, name="bell")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    artifact = compile_qiskit_circuit(
        circuit,
        options=CompilationOptions(target_basis=NATIVE_BASIS, seed=19),
    )

    assert "CZ Q0 Q1" in artifact.qcis
    assert artifact.qcis.count("M Q") == 2
    assert [(item.physical_qubit, item.classical_bit) for item in artifact.measurements] == [
        (0, 0),
        (1, 1),
    ]
    allowed = set(NATIVE_BASIS) | {"measure_bit"}
    assert set(operation_names(artifact.circuit)) <= allowed


def test_supported_cqlib_circuit_converts_back_to_qiskit() -> None:
    source = QuantumCircuit(2, 2, name="round-trip")
    source.h(0)
    source.cx(0, 1)
    source.measure([0, 1], [0, 1])
    bundle = qiskit_to_cqlib(source)

    restored = cqlib_to_qiskit(bundle.circuit, metadata=bundle.metadata)

    assert restored.name == "round-trip"
    assert [item.operation.name for item in restored.data] == ["h", "cx", "measure", "measure"]
    assert restored.num_qubits == 2
    assert restored.num_clbits == 2


def test_every_declared_m2_gate_converts_in_both_directions() -> None:
    circuit = QuantumCircuit(3, 3, name="all-supported-gates")
    circuit.id(0)
    circuit.h(0)
    circuit.x(0)
    circuit.y(0)
    circuit.z(0)
    circuit.s(0)
    circuit.sdg(0)
    circuit.t(0)
    circuit.tdg(0)
    circuit.append(X2PGate(), [0])
    circuit.rx(0.1, 0)
    circuit.ry(0.2, 0)
    circuit.rz(0.3, 0)
    circuit.p(0.4, 0)
    circuit.u(0.1, 0.2, 0.3, 0)
    circuit.append(XYGate(0.5), [0])
    circuit.append(XY2PGate(0.6), [0])
    circuit.append(XY2MGate(0.7), [0])
    circuit.append(RXYGate(0.8, 0.9), [0])
    circuit.cx(0, 1)
    circuit.cy(0, 1)
    circuit.cz(0, 1)
    circuit.swap(0, 1)
    circuit.rxx(0.1, 0, 1)
    circuit.ryy(0.2, 0, 1)
    circuit.rzz(0.3, 0, 1)
    circuit.rzx(0.4, 0, 1)
    circuit.crx(0.5, 0, 1)
    circuit.cry(0.6, 0, 1)
    circuit.crz(0.7, 0, 1)
    circuit.append(FSimGate(0.8, 0.9), [0, 1])
    circuit.ccx(0, 1, 2)
    circuit.reset(2)
    circuit.barrier(0, 1, 2)
    circuit.measure([0, 1, 2], [0, 1, 2])

    bundle = qiskit_to_cqlib(circuit)
    restored = cqlib_to_qiskit(bundle.circuit, metadata=bundle.metadata)

    expected = [item.operation.name for item in circuit.data]
    assert [item.operation.name for item in restored.data] == expected
    assert restored.num_clbits == 3


def test_global_phase_is_preserved_in_both_directions() -> None:
    circuit = QuantumCircuit(1, 1, global_phase=pi / 3)
    circuit.x(0)
    circuit.measure(0, 0)

    bundle = qiskit_to_cqlib(circuit)
    restored = cqlib_to_qiskit(bundle.circuit, metadata=bundle.metadata)

    assert bundle.metadata.global_phase == pytest.approx(pi / 3)
    assert float(restored.global_phase) == pytest.approx(pi / 3)


def test_unbound_qiskit_parameters_are_rejected() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit(1, 1)
    circuit.rx(theta, 0)
    circuit.measure(0, 0)

    with pytest.raises(AdapterConversionError, match="unbound parameters"):
        qiskit_to_cqlib(circuit)


def test_instruction_after_measurement_is_rejected() -> None:
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    circuit.x(0)

    with pytest.raises(AdapterConversionError, match="after a measurement"):
        qiskit_to_cqlib(circuit)


def test_unknown_qiskit_instruction_is_rejected() -> None:
    circuit = QuantumCircuit(1, 1)
    circuit.append(Gate("unknown", 1, []), [0])
    circuit.measure(0, 0)

    with pytest.raises(AdapterConversionError, match="unsupported Qiskit instruction"):
        qiskit_to_cqlib(circuit)


def test_two_measurements_to_same_classical_bit_are_rejected() -> None:
    circuit = QuantumCircuit(2, 1)
    circuit.measure(0, 0)
    circuit.measure(1, 0)

    with pytest.raises(AdapterConversionError, match="at most one final measurement"):
        qiskit_to_cqlib(circuit)


def test_wrong_input_type_and_empty_circuit_are_rejected() -> None:
    with pytest.raises(TypeError, match="QuantumCircuit"):
        qiskit_to_cqlib(object())  # type: ignore[arg-type]
    with pytest.raises(AdapterConversionError, match="at least one qubit"):
        qiskit_to_cqlib(QuantumCircuit())


def test_reverse_delay_conversion_has_explicit_m2_error() -> None:
    from cqlib import Circuit

    circuit = Circuit(1)
    circuit.delay(0, 1.0)

    with pytest.raises(AdapterConversionError, match="delay conversion"):
        cqlib_to_qiskit(circuit)
