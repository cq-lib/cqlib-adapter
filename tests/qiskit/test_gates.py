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

from math import pi

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("cqlib")
from cqlib.circuit.gates import RXY as CqlibRXY
from cqlib.circuit.gates import XY as CqlibXY
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator

from cqlib_adapter.qiskit import (
    FSimGate,
    RXYGate,
    X2MGate,
    X2PGate,
    XY2MGate,
    XY2PGate,
    XYGate,
    Y2MGate,
    Y2PGate,
    qcis_instruction,
    qcis_name_mapping,
)

pytestmark = pytest.mark.qiskit


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        (X2PGate(), X2MGate()),
        (Y2PGate(), Y2MGate()),
        (XY2PGate(0.3), XY2MGate(0.3)),
    ],
)
def test_half_rotation_inverse_pairs(positive: object, negative: object) -> None:
    assert positive.inverse().name == negative.name  # type: ignore[attr-defined]
    assert negative.inverse().name == positive.name  # type: ignore[attr-defined]
    identity = Operator(positive).compose(Operator(negative))
    assert identity.equiv(Operator.from_label("I"))


@pytest.mark.parametrize("axis", [0.0, 0.31, -0.29, pi / 2])
def test_xy_gate_matrix_matches_cqlib(axis: float) -> None:
    np.testing.assert_allclose(
        Operator(XYGate(axis)).data,
        CqlibXY.matrix([axis]),
        atol=1e-12,
    )


def test_xy_gate_inverse_and_symbolic_definition() -> None:
    axis = 0.31
    gate = XYGate(axis)
    assert gate.inverse().params == pytest.approx([axis + pi])
    assert Operator(gate.inverse()).compose(Operator(gate)).equiv(Operator.from_label("I"))

    parameter = Parameter("axis")
    circuit = QuantumCircuit(1)
    circuit.append(XYGate(parameter), [0])
    bound = circuit.assign_parameters({parameter: axis})
    decomposed = transpile(bound, basis_gates=["rz", "rx"], optimization_level=0)

    assert set(decomposed.count_ops()).issubset({"rz", "rx"})
    np.testing.assert_allclose(
        Operator(decomposed).data,
        CqlibXY.matrix([axis]),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("theta", "phi"),
    [(0.37, -0.29), (pi, 0.31), (-0.42, pi / 3)],
)
def test_rxy_gate_matrix_matches_cqlib(theta: float, phi: float) -> None:
    np.testing.assert_allclose(
        Operator(RXYGate(theta, phi)).data,
        CqlibRXY.matrix([theta, phi]),
        atol=1e-12,
    )


def test_custom_gates_can_be_appended_and_parameter_bound() -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = QuantumCircuit(2)
    circuit.append(XYGate(theta), [0])
    circuit.append(RXYGate(theta, phi), [0])
    circuit.append(FSimGate(theta, phi), [0, 1])

    bound = circuit.assign_parameters({theta: pi / 4, phi: pi / 7})

    assert not bound.parameters
    assert [item.operation.name for item in bound.data] == ["xy", "rxy", "fsim"]
    assert list(bound.data[1].operation.params) == pytest.approx([pi / 4, pi / 7])


def test_qcis_name_mapping_contains_standard_and_custom_native_gates() -> None:
    mapping = qcis_name_mapping()

    assert mapping["rz"].name == "rz"
    assert mapping["cz"].name == "cz"
    assert mapping["x2p"].name == "x2p"
    assert mapping["rxy"].num_qubits == 1
    assert mapping["fsim"].num_qubits == 2


@pytest.mark.parametrize("name", ["X2P", "XY2M", "RXY", "FSIM", "CZ", "RZ"])
def test_qcis_instruction_normalizes_case(name: str) -> None:
    assert qcis_instruction(name.lower()).name == name.lower()


def test_unknown_qcis_gate_is_rejected() -> None:
    with pytest.raises(KeyError, match="unsupported QCIS gate"):
        qcis_instruction("not-a-gate")
