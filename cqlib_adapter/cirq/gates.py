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

"""Cirq representations of QCIS-native gates."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any, cast

import cirq
import numpy as np


def _parameterized(*values: Any) -> bool:
    return any(cirq.is_parameterized(value) for value in values)


def _parameter_names(*values: Any) -> set[str]:
    names: set[str] = set()
    for value in values:
        names.update(cirq.parameter_names(value))
    return names


def _rxy_matrix(theta: float, phi: float) -> np.ndarray:
    half = theta / 2
    phase = np.exp(1j * phi)
    return np.asarray(
        [
            [cos(half), -1j * np.conjugate(phase) * sin(half)],
            [-1j * phase * sin(half), cos(half)],
        ],
        dtype=np.complex128,
    )


class _FixedHalfRotation(cirq.Gate):
    axis = "X"
    angle = pi / 2
    label = "X2P"

    def _num_qubits_(self) -> int:
        return 1

    def _unitary_(self) -> np.ndarray:
        gate = cirq.rx(self.angle) if self.axis == "X" else cirq.ry(self.angle)
        return cirq.unitary(gate)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        del args
        return self.label

    def __repr__(self) -> str:
        return f"cqlib_adapter.cirq.{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))


class X2PGate(_FixedHalfRotation):
    """Positive half-X rotation, RX(pi/2)."""

    label = "X2P"

    def __pow__(self, exponent: Any) -> Any:
        if exponent == -1:
            return X2MGate()
        if exponent == 1:
            return self
        return NotImplemented


class X2MGate(_FixedHalfRotation):
    """Negative half-X rotation, RX(-pi/2)."""

    angle = -pi / 2
    label = "X2M"

    def __pow__(self, exponent: Any) -> Any:
        if exponent == -1:
            return X2PGate()
        if exponent == 1:
            return self
        return NotImplemented


class Y2PGate(_FixedHalfRotation):
    """Positive half-Y rotation, RY(pi/2)."""

    axis = "Y"
    label = "Y2P"

    def __pow__(self, exponent: Any) -> Any:
        if exponent == -1:
            return Y2MGate()
        if exponent == 1:
            return self
        return NotImplemented


class Y2MGate(_FixedHalfRotation):
    """Negative half-Y rotation, RY(-pi/2)."""

    axis = "Y"
    angle = -pi / 2
    label = "Y2M"

    def __pow__(self, exponent: Any) -> Any:
        if exponent == -1:
            return Y2PGate()
        if exponent == 1:
            return self
        return NotImplemented


@dataclass(frozen=True)
class XYGate(cirq.Gate):
    """One-qubit native XY(theta) pulse."""

    theta: Any

    def _num_qubits_(self) -> int:
        return 1

    def _is_parameterized_(self) -> bool:
        return _parameterized(self.theta)

    def _parameter_names_(self) -> set[str]:
        return _parameter_names(self.theta)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> XYGate:
        return XYGate(resolver.value_of(self.theta, recursive))

    def _unitary_(self) -> Any:
        if self._is_parameterized_():
            return NotImplemented
        return cirq.unitary(cirq.rx(float(self.theta)))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        del args
        return f"XY({self.theta})"

    def __pow__(self, exponent: Any) -> Any:
        if isinstance(exponent, int | float):
            return XYGate(self.theta * exponent)
        return NotImplemented


@dataclass(frozen=True)
class XY2PGate(cirq.Gate):
    """Positive half-pi pulse around the XY-plane axis at angle phi."""

    phi: Any

    def _num_qubits_(self) -> int:
        return 1

    def _is_parameterized_(self) -> bool:
        return _parameterized(self.phi)

    def _parameter_names_(self) -> set[str]:
        return _parameter_names(self.phi)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> XY2PGate:
        return XY2PGate(resolver.value_of(self.phi, recursive))

    def _unitary_(self) -> Any:
        if self._is_parameterized_():
            return NotImplemented
        return _rxy_matrix(pi / 2, float(self.phi))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        del args
        return f"XY2P({self.phi})"

    def __pow__(self, exponent: Any) -> Any:
        if exponent == -1:
            return XY2MGate(self.phi)
        if exponent == 1:
            return self
        return NotImplemented


@dataclass(frozen=True)
class XY2MGate(cirq.Gate):
    """Negative half-pi pulse around the XY-plane axis at angle phi."""

    phi: Any

    def _num_qubits_(self) -> int:
        return 1

    def _is_parameterized_(self) -> bool:
        return _parameterized(self.phi)

    def _parameter_names_(self) -> set[str]:
        return _parameter_names(self.phi)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> XY2MGate:
        return XY2MGate(resolver.value_of(self.phi, recursive))

    def _unitary_(self) -> Any:
        if self._is_parameterized_():
            return NotImplemented
        return _rxy_matrix(-pi / 2, float(self.phi))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        del args
        return f"XY2M({self.phi})"

    def __pow__(self, exponent: Any) -> Any:
        if exponent == -1:
            return XY2PGate(self.phi)
        if exponent == 1:
            return self
        return NotImplemented


@dataclass(frozen=True)
class RXYGate(cirq.Gate):
    """Rotation by theta around an XY-plane axis at angle phi."""

    theta: Any
    phi: Any

    def _num_qubits_(self) -> int:
        return 1

    def _is_parameterized_(self) -> bool:
        return _parameterized(self.theta, self.phi)

    def _parameter_names_(self) -> set[str]:
        return _parameter_names(self.theta, self.phi)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> RXYGate:
        return RXYGate(
            resolver.value_of(self.theta, recursive),
            resolver.value_of(self.phi, recursive),
        )

    def _unitary_(self) -> Any:
        if self._is_parameterized_():
            return NotImplemented
        return _rxy_matrix(float(self.theta), float(self.phi))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        del args
        return f"RXY({self.theta},{self.phi})"

    def __pow__(self, exponent: Any) -> Any:
        if isinstance(exponent, int | float):
            return RXYGate(self.theta * exponent, self.phi)
        return NotImplemented


FSimGate = cirq.FSimGate


QCIS_GATE_TYPES = {
    "X2P": X2PGate,
    "X2M": X2MGate,
    "Y2P": Y2PGate,
    "Y2M": Y2MGate,
    "XY": XYGate,
    "XY2P": XY2PGate,
    "XY2M": XY2MGate,
    "RXY": RXYGate,
    "FSIM": FSimGate,
}


def qcis_gate(name: str, *params: Any) -> cirq.Gate:
    """Create one Cirq gate from a QCIS/cqlib instruction name."""

    normalized = name.strip().upper()
    fixed: dict[str, cirq.Gate] = {
        "I": cirq.I,
        "H": cirq.H,
        "X": cirq.X,
        "Y": cirq.Y,
        "Z": cirq.Z,
        "S": cirq.S,
        "SDG": cirq.S**-1,
        "T": cirq.T,
        "TDG": cirq.T**-1,
        "CX": cirq.CNOT,
        "CNOT": cirq.CNOT,
        "CZ": cirq.CZ,
        "SWAP": cirq.SWAP,
        "CCX": cirq.TOFFOLI,
        "X2P": X2PGate(),
        "X2M": X2MGate(),
        "Y2P": Y2PGate(),
        "Y2M": Y2MGate(),
    }
    if normalized in fixed:
        if params:
            raise ValueError(f"QCIS gate {normalized} takes no parameters")
        return fixed[normalized]
    parameterized: dict[str, tuple[type[Any], int]] = {
        "XY": (XYGate, 1),
        "XY2P": (XY2PGate, 1),
        "XY2M": (XY2MGate, 1),
        "RXY": (RXYGate, 2),
        "FSIM": (FSimGate, 2),
    }
    if normalized == "GPHASE":
        if len(params) != 1:
            raise ValueError("QCIS gate GPHASE requires one parameter")
        if cirq.is_parameterized(params[0]):
            raise ValueError("QCIS GPHASE factory requires a numeric phase")
        return cirq.GlobalPhaseGate(np.exp(1j * float(params[0])))
    if normalized in {"RX", "RY", "RZ"}:
        if len(params) != 1:
            raise ValueError(f"QCIS gate {normalized} requires one parameter")
        return {"RX": cirq.rx, "RY": cirq.ry, "RZ": cirq.rz}[normalized](params[0])
    try:
        gate_type, count = parameterized[normalized]
    except KeyError as exc:
        raise KeyError(f"unsupported QCIS gate {name!r}") from exc
    if len(params) != count:
        raise ValueError(f"QCIS gate {normalized} requires {count} parameters")
    return cast(cirq.Gate, gate_type(*params))


__all__ = [
    "QCIS_GATE_TYPES",
    "FSimGate",
    "RXYGate",
    "X2MGate",
    "X2PGate",
    "XY2MGate",
    "XY2PGate",
    "XYGate",
    "Y2MGate",
    "Y2PGate",
    "qcis_gate",
]
