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

"""Qiskit representations of QCIS-native gates.

These are ordinary Qiskit Gate objects that can be appended to a
QuantumCircuit, advertised by a Target and translated directly to cqlib.
"""

from __future__ import annotations

from math import pi
from typing import Any

from qiskit.circuit import Gate, Parameter, QuantumCircuit
from qiskit.circuit.library import (
    CCXGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CXGate,
    CYGate,
    CZGate,
    HGate,
    IGate,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    TdgGate,
    TGate,
    UGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.circuit.parameterexpression import ParameterValueType


class X2PGate(Gate):
    """Positive half-X rotation, RX(pi / 2)."""

    def __init__(self, label: str | None = None) -> None:
        super().__init__("x2p", 1, [], label=label)

    def _define(self) -> None:
        definition = QuantumCircuit(1)
        definition.rx(pi / 2, 0)
        self.definition = definition

    def inverse(self, annotated: bool = False) -> X2MGate:
        del annotated
        return X2MGate(label=self.label)


class X2MGate(Gate):
    """Negative half-X rotation, RX(-pi / 2)."""

    def __init__(self, label: str | None = None) -> None:
        super().__init__("x2m", 1, [], label=label)

    def _define(self) -> None:
        definition = QuantumCircuit(1)
        definition.rx(-pi / 2, 0)
        self.definition = definition

    def inverse(self, annotated: bool = False) -> X2PGate:
        del annotated
        return X2PGate(label=self.label)


class Y2PGate(Gate):
    """Positive half-Y rotation, RY(pi / 2)."""

    def __init__(self, label: str | None = None) -> None:
        super().__init__("y2p", 1, [], label=label)

    def _define(self) -> None:
        definition = QuantumCircuit(1)
        definition.ry(pi / 2, 0)
        self.definition = definition

    def inverse(self, annotated: bool = False) -> Y2MGate:
        del annotated
        return Y2MGate(label=self.label)


class Y2MGate(Gate):
    """Negative half-Y rotation, RY(-pi / 2)."""

    def __init__(self, label: str | None = None) -> None:
        super().__init__("y2m", 1, [], label=label)

    def _define(self) -> None:
        definition = QuantumCircuit(1)
        definition.ry(-pi / 2, 0)
        self.definition = definition

    def inverse(self, annotated: bool = False) -> Y2PGate:
        del annotated
        return Y2PGate(label=self.label)


class XYGate(Gate):
    """Pi rotation about an XY-plane axis whose phase is ``theta``."""

    def __init__(self, theta: ParameterValueType, label: str | None = None) -> None:
        super().__init__("xy", 1, [theta], label=label)

    def _define(self) -> None:
        theta = self.params[0]
        definition = QuantumCircuit(1)
        definition.rz(-theta, 0)
        definition.rx(pi, 0)
        definition.rz(theta, 0)
        self.definition = definition

    def inverse(self, annotated: bool = False) -> XYGate:
        del annotated
        return XYGate(self.params[0] + pi, label=self.label)


class XY2PGate(Gate):
    """Positive half-pi XY-plane pulse with axis angle phi."""

    def __init__(self, phi: ParameterValueType, label: str | None = None) -> None:
        super().__init__("xy2p", 1, [phi], label=label)

    def _define(self) -> None:
        phi = self.params[0]
        definition = QuantumCircuit(1)
        definition.rz(pi / 2 - phi, 0)
        definition.ry(pi / 2, 0)
        definition.rz(phi - pi / 2, 0)
        self.definition = definition

    def inverse(self, annotated: bool = False) -> XY2MGate:
        del annotated
        return XY2MGate(self.params[0], label=self.label)


class XY2MGate(Gate):
    """Negative half-pi XY-plane pulse with axis angle phi."""

    def __init__(self, phi: ParameterValueType, label: str | None = None) -> None:
        super().__init__("xy2m", 1, [phi], label=label)

    def _define(self) -> None:
        phi = self.params[0]
        definition = QuantumCircuit(1)
        definition.rz(-pi / 2 - phi, 0)
        definition.ry(pi / 2, 0)
        definition.rz(phi + pi / 2, 0)
        self.definition = definition

    def inverse(self, annotated: bool = False) -> XY2PGate:
        del annotated
        return XY2PGate(self.params[0], label=self.label)


class RXYGate(Gate):
    """Rotation by theta about an axis at phi in the XY plane."""

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        label: str | None = None,
    ) -> None:
        super().__init__("rxy", 1, [theta, phi], label=label)

    def _define(self) -> None:
        theta, phi = self.params
        definition = QuantumCircuit(1)
        definition.rz(-phi, 0)
        definition.rx(theta, 0)
        definition.rz(phi, 0)
        self.definition = definition

    def inverse(self, annotated: bool = False) -> RXYGate:
        del annotated
        return RXYGate(-self.params[0], self.params[1], label=self.label)


class FSimGate(Gate):
    """Two-qubit native FSIM(theta, phi) gate."""

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        label: str | None = None,
    ) -> None:
        super().__init__("fsim", 2, [theta, phi], label=label)

    def inverse(self, annotated: bool = False) -> FSimGate:
        del annotated
        return FSimGate(-self.params[0], -self.params[1], label=self.label)


def qcis_instruction(name: str) -> Any:
    """Return a Qiskit instruction for one cqlib/QCIS gate name."""

    theta = Parameter("theta")
    phi = Parameter("phi")
    mapping: dict[str, Any] = {
        "I": IGate(),
        "H": HGate(),
        "X": XGate(),
        "Y": YGate(),
        "Z": ZGate(),
        "S": SGate(),
        "SDG": SdgGate(),
        "T": TGate(),
        "TDG": TdgGate(),
        "RX": RXGate(theta),
        "RY": RYGate(theta),
        "RZ": RZGate(theta),
        "PHASE": PhaseGate(theta),
        "U": UGate(theta, phi, Parameter("lambda")),
        "CX": CXGate(),
        "CY": CYGate(),
        "CZ": CZGate(),
        "CCX": CCXGate(),
        "SWAP": SwapGate(),
        "CRX": CRXGate(theta),
        "CRY": CRYGate(theta),
        "CRZ": CRZGate(theta),
        "RXX": RXXGate(theta),
        "RYY": RYYGate(theta),
        "RZZ": RZZGate(theta),
        "RZX": RZXGate(theta),
        "RXY": RXYGate(theta, phi),
        "XY": XYGate(theta),
        "X2P": X2PGate(),
        "X2M": X2MGate(),
        "Y2P": Y2PGate(),
        "Y2M": Y2MGate(),
        "XY2P": XY2PGate(phi),
        "XY2M": XY2MGate(phi),
        "FSIM": FSimGate(theta, phi),
    }
    normalized = name.strip().upper()
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise KeyError(f"unsupported QCIS gate {name!r}") from exc


def qcis_name_mapping() -> dict[str, Any]:
    """Return Qiskit names for all supported cqlib/QCIS gates."""

    names = (
        "I",
        "H",
        "X",
        "Y",
        "Z",
        "S",
        "SDG",
        "T",
        "TDG",
        "RX",
        "RY",
        "RZ",
        "PHASE",
        "U",
        "CX",
        "CY",
        "CZ",
        "CCX",
        "SWAP",
        "CRX",
        "CRY",
        "CRZ",
        "RXX",
        "RYY",
        "RZZ",
        "RZX",
        "RXY",
        "XY",
        "X2P",
        "X2M",
        "Y2P",
        "Y2M",
        "XY2P",
        "XY2M",
        "FSIM",
    )
    return {name.lower(): qcis_instruction(name) for name in names}


def _register_equivalences() -> None:
    """Teach Qiskit's basis translator the Tianyan half-rotation basis."""

    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

    h_definition = QuantumCircuit(1)
    h_definition.rz(pi, 0)
    h_definition.append(Y2PGate(), [0])
    h_definition.global_phase = pi / 2
    SessionEquivalenceLibrary.add_equivalence(HGate(), h_definition)

    cx_definition = QuantumCircuit(2)
    cx_definition.h(1)
    cx_definition.cz(0, 1)
    cx_definition.h(1)
    SessionEquivalenceLibrary.add_equivalence(CXGate(), cx_definition)

    x_definition = QuantumCircuit(1)
    x_definition.append(X2PGate(), [0])
    x_definition.append(X2PGate(), [0])
    x_definition.global_phase = pi / 2
    SessionEquivalenceLibrary.add_equivalence(XGate(), x_definition)

    y_definition = QuantumCircuit(1)
    y_definition.append(Y2PGate(), [0])
    y_definition.append(Y2PGate(), [0])
    y_definition.global_phase = pi / 2
    SessionEquivalenceLibrary.add_equivalence(YGate(), y_definition)

    theta = Parameter("theta")
    rx_definition = QuantumCircuit(1)
    rx_definition.rz(pi / 2, 0)
    rx_definition.append(X2PGate(), [0])
    rx_definition.rz(theta, 0)
    rx_definition.append(X2MGate(), [0])
    rx_definition.rz(-pi / 2, 0)
    SessionEquivalenceLibrary.add_equivalence(RXGate(theta), rx_definition)

    ry_definition = QuantumCircuit(1)
    ry_definition.append(X2PGate(), [0])
    ry_definition.rz(theta, 0)
    ry_definition.append(X2MGate(), [0])
    SessionEquivalenceLibrary.add_equivalence(RYGate(theta), ry_definition)


_register_equivalences()


__all__ = [
    "FSimGate",
    "RXYGate",
    "X2MGate",
    "X2PGate",
    "XY2MGate",
    "XY2PGate",
    "XYGate",
    "Y2MGate",
    "Y2PGate",
    "qcis_instruction",
    "qcis_name_mapping",
]
