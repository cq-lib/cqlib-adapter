from __future__ import annotations

from math import pi

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("cqlib")
from qiskit import QuantumCircuit
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
