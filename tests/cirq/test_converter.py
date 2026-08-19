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

import cirq
import numpy as np
import pytest
import sympy
from cqlib.device import Layout

from cqlib_adapter.cirq import (
    X2PGate,
    XYGate,
    cirq_to_cqlib,
    compile_cirq_circuit,
    cqlib_to_cirq,
)
from cqlib_adapter.cirq.testing import MockCloudBackend
from cqlib_adapter.common import AdapterConversionError, CompilationOptions, NormalizedDevice

pytestmark = pytest.mark.cirq


def test_cirq_circuit_translates_qubits_native_gates_and_measurement_keys() -> None:
    a, b, c = cirq.NamedQubit.range(3, prefix="q")
    circuit = cirq.Circuit(
        cirq.H(a),
        X2PGate().on(b),
        cirq.CNOT(b, c),
        cirq.measure(a, key="first"),
        cirq.measure(b, c, key="pair", invert_mask=(False, True)),
    )

    bundle = cirq_to_cqlib(circuit, qubit_order=(a, b, c))

    assert bundle.metadata.framework == "cirq"
    assert bundle.metadata.measurements.register_sizes == {"first": 1, "pair": 2}
    assert bundle.metadata.extras["measurement_key_bits"] == (
        ("first", (0,)),
        ("pair", (1, 2)),
    )
    text = [str(operation) for operation in bundle.circuit.operations]
    assert text[:3] == ["H Q0", "X2P Q1", "CX Q1 Q2"]
    assert "X Q2" in text
    assert [item for item in text if item.startswith("measure_bit")] == [
        "measure_bit Q0",
        "measure_bit Q1",
        "measure_bit Q2",
    ]


def test_compile_uses_real_cqlib_native_lowering_and_initial_layout() -> None:
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(*qubits, key="state"),
    )
    device = NormalizedDevice.from_backend(MockCloudBackend([], size=3))
    layout = Layout.from_pairs([(0, 2), (1, 1), (2, 0)], physical_count=3)

    artifact = compile_cirq_circuit(
        circuit,
        device=device,
        options=CompilationOptions(initial_layout=layout, seed=17),
    )

    assert artifact.circuit.validate() is None
    assert "CZ Q2 Q1" in artifact.qcis or "CZ Q1 Q2" in artifact.qcis
    assert tuple(item.physical_qubit for item in artifact.measurements) == (2, 1, 0)
    assert artifact.qcis.count("M Q") == 3


def test_real_cqlib_topology_routes_nonadjacent_cirq_operation() -> None:
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CNOT(qubits[0], qubits[2]), cirq.measure(*qubits, key="m"))
    device = NormalizedDevice.from_backend(MockCloudBackend([], size=3))

    artifact = compile_cirq_circuit(circuit, device=device)

    assert any(step.name == "route.sabre" and not step.skipped for step in artifact.steps)
    for operation in artifact.circuit.operations:
        indices = tuple(qubit.index for qubit in operation.qubits)
        if len(indices) == 2:
            assert device.supports_coupling(*indices, either_direction=True)


def test_cqlib_roundtrip_preserves_supported_operations_and_keys() -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.global_phase_operation(1j),
        cirq.X(qubits[0]),
        cirq.CZ(*qubits),
        cirq.measure(*qubits, key="answer"),
    )
    bundle = cirq_to_cqlib(circuit)

    restored = cqlib_to_cirq(bundle.circuit, metadata=bundle.metadata)

    assert cirq.measurement_key_names(restored) == frozenset({"answer"})
    assert any(operation.gate == cirq.CZ for operation in restored.all_operations())
    assert any(
        isinstance(operation.gate, cirq.GlobalPhaseGate) for operation in restored.all_operations()
    )
    assert tuple(restored.all_qubits()) == tuple(qubits)


def test_xy_roundtrip_preserves_axis_phase_and_gate_semantics() -> None:
    axis = 0.31
    qubit = cirq.LineQubit(0)
    source = cirq.Circuit(XYGate(axis).on(qubit), cirq.measure(qubit, key="result"))

    bundle = cirq_to_cqlib(source)
    restored = cqlib_to_cirq(bundle.circuit, metadata=bundle.metadata)
    restored_xy = [
        operation.gate
        for operation in restored.all_operations()
        if isinstance(operation.gate, XYGate)
    ]

    converted_xy = bundle.circuit.operations[0]
    assert converted_xy.instruction.instruction.name.lower() == "xy"
    assert converted_xy.params == pytest.approx([axis])
    assert restored_xy == [XYGate(axis)]
    np.testing.assert_allclose(
        cirq.unitary(restored_xy[0]),
        cirq.unitary(XYGate(axis)),
        atol=1e-12,
    )


def test_parameter_and_measurement_boundary_errors_are_explicit() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    theta = sympy.Symbol("theta")
    with pytest.raises(AdapterConversionError, match="unbound parameters"):
        cirq_to_cqlib(cirq.Circuit(cirq.rx(theta)(q0), cirq.measure(q0)))
    with pytest.raises(AdapterConversionError, match="repeated Cirq measurement key"):
        cirq_to_cqlib(
            cirq.Circuit(
                cirq.measure(q0, key="same"),
                cirq.measure(q1, key="same"),
            )
        )
    with pytest.raises(AdapterConversionError, match="occurs after measurement"):
        cirq_to_cqlib(
            cirq.Circuit(
                cirq.Moment([cirq.measure(q0, key="m")]),
                cirq.Moment([cirq.X(q1)]),
            )
        )
    with pytest.raises(AdapterConversionError, match="final measurement"):
        cirq_to_cqlib(cirq.Circuit(cirq.X(q0)))
