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

"""PennyLane representations of QCIS-native operations."""

from __future__ import annotations

from math import pi
from typing import Any

import numpy as np
import pennylane as qml
from pennylane.operation import Operation


def _stack_2x2(a: Any, b: Any, c: Any, d: Any) -> Any:
    return qml.math.stack(
        [qml.math.stack([a, b], axis=-1), qml.math.stack([c, d], axis=-1)],
        axis=-2,
    )


class X2P(Operation):
    """Positive half-X rotation, RX(pi / 2)."""

    num_wires = 1
    num_params = 0
    grad_method = None

    @staticmethod
    def compute_matrix() -> Any:
        return qml.RX.compute_matrix(pi / 2)

    def adjoint(self) -> X2M:
        return X2M(wires=self.wires)


class X2M(Operation):
    """Negative half-X rotation, RX(-pi / 2)."""

    num_wires = 1
    num_params = 0
    grad_method = None

    @staticmethod
    def compute_matrix() -> Any:
        return qml.RX.compute_matrix(-pi / 2)

    def adjoint(self) -> X2P:
        return X2P(wires=self.wires)


class Y2P(Operation):
    """Positive half-Y rotation, RY(pi / 2)."""

    num_wires = 1
    num_params = 0
    grad_method = None

    @staticmethod
    def compute_matrix() -> Any:
        return qml.RY.compute_matrix(pi / 2)

    def adjoint(self) -> Y2M:
        return Y2M(wires=self.wires)


class Y2M(Operation):
    """Negative half-Y rotation, RY(-pi / 2)."""

    num_wires = 1
    num_params = 0
    grad_method = None

    @staticmethod
    def compute_matrix() -> Any:
        return qml.RY.compute_matrix(-pi / 2)

    def adjoint(self) -> Y2P:
        return Y2P(wires=self.wires)


class XY(Operation):
    """One-qubit native XY(theta) pulse."""

    num_wires = 1
    num_params = 1
    ndim_params = (0,)
    grad_method = None

    @staticmethod
    def compute_matrix(theta: Any) -> Any:
        return qml.RX.compute_matrix(theta)

    def adjoint(self) -> XY:
        return XY(-self.data[0], wires=self.wires)


class XY2P(Operation):
    """Positive half-pi pulse about an XY-plane axis at angle phi."""

    num_wires = 1
    num_params = 1
    ndim_params = (0,)
    grad_method = None

    @staticmethod
    def compute_matrix(phi: Any) -> Any:
        phase = qml.math.exp(1j * phi)
        one = qml.math.ones_like(phase)
        scale = qml.math.cast_like(1 / np.sqrt(2), phase)
        return scale * _stack_2x2(one, -1j / phase, -1j * phase, one)

    def adjoint(self) -> XY2M:
        return XY2M(self.data[0], wires=self.wires)


class XY2M(Operation):
    """Negative half-pi pulse about an XY-plane axis at angle phi."""

    num_wires = 1
    num_params = 1
    ndim_params = (0,)
    grad_method = None

    @staticmethod
    def compute_matrix(phi: Any) -> Any:
        phase = qml.math.exp(1j * phi)
        one = qml.math.ones_like(phase)
        scale = qml.math.cast_like(1 / np.sqrt(2), phase)
        return scale * _stack_2x2(one, 1j / phase, 1j * phase, one)

    def adjoint(self) -> XY2P:
        return XY2P(self.data[0], wires=self.wires)


class RXY(Operation):
    """Rotation by theta about an XY-plane axis at angle phi."""

    num_wires = 1
    num_params = 2
    ndim_params = (0, 0)
    grad_method = None

    @staticmethod
    def compute_matrix(theta: Any, phi: Any) -> Any:
        left = qml.RZ.compute_matrix(phi)
        middle = qml.RX.compute_matrix(theta)
        right = qml.RZ.compute_matrix(-phi)
        return qml.math.matmul(qml.math.matmul(left, middle), right)

    def adjoint(self) -> RXY:
        return RXY(-self.data[0], self.data[1], wires=self.wires)


class FSim(Operation):
    """Two-qubit native fSim(theta, phi) operation."""

    num_wires = 2
    num_params = 2
    ndim_params = (0, 0)
    grad_method = None

    @staticmethod
    def compute_matrix(theta: Any, phi: Any) -> Any:
        cosine = qml.math.cos(theta)
        sine = qml.math.sin(theta)
        phase = qml.math.exp(-1j * phi)
        one = qml.math.ones_like(phase)
        zero = qml.math.zeros_like(phase)
        cosine = qml.math.cast_like(cosine, phase)
        sine = qml.math.cast_like(sine, phase)
        return qml.math.stack(
            [
                qml.math.stack([one, zero, zero, zero]),
                qml.math.stack([zero, cosine, -1j * sine, zero]),
                qml.math.stack([zero, -1j * sine, cosine, zero]),
                qml.math.stack([zero, zero, zero, phase]),
            ]
        )

    def adjoint(self) -> FSim:
        return FSim(-self.data[0], -self.data[1], wires=self.wires)


# Compatibility aliases matching the Qiskit adapter and the first adapter version.
X2PGate = X2P
X2MGate = X2M
Y2PGate = Y2P
Y2MGate = Y2M
XYGate = XY
XY2PGate = XY2P
XY2MGate = XY2M
RXYGate = RXY
FSimGate = FSim

QCIS_OPERATION_TYPES = {
    "X2P": X2P,
    "X2M": X2M,
    "Y2P": Y2P,
    "Y2M": Y2M,
    "XY": XY,
    "XY2P": XY2P,
    "XY2M": XY2M,
    "RXY": RXY,
    "FSIM": FSim,
}


__all__ = [
    "QCIS_OPERATION_TYPES",
    "RXY",
    "X2M",
    "X2P",
    "XY",
    "XY2M",
    "XY2P",
    "Y2M",
    "Y2P",
    "FSim",
    "FSimGate",
    "RXYGate",
    "X2MGate",
    "X2PGate",
    "XY2MGate",
    "XY2PGate",
    "XYGate",
    "Y2MGate",
    "Y2PGate",
]
