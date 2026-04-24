# This code is part of cqlib.
#
# Copyright (C) 2026 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pennylane as qml
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike
import numpy as np

def _stack_matrix(elements):
    return qml.math.stack([
        qml.math.stack(elements[:2], axis=-1),
        qml.math.stack(elements[2:], axis=-1)
    ], axis=-2)


class X2PGate(Operation):
    r"""Native X rotation by +pi/2 (Sqrt X)."""
    num_wires = 1
    num_params = 0
    grad_method = None 

    @staticmethod
    def compute_matrix() -> TensorLike:
        # 1/sqrt(2) * [[1, -i], [-i, 1]]
        isqrt2 = 1 / np.sqrt(2)
        return qml.math.array([
            [isqrt2, -1j * isqrt2],
            [-1j * isqrt2, isqrt2]
        ], dtype=complex)

    def adjoint(self):
        return X2MGate(wires=self.wires)

class X2MGate(Operation):
    r"""Native X rotation by -pi/2 (Sqrt X dag)."""
    num_wires = 1
    num_params = 0
    grad_method = None

    @staticmethod
    def compute_matrix() -> TensorLike:
        isqrt2 = 1 / np.sqrt(2)
        return qml.math.array([
            [isqrt2, 1j * isqrt2],
            [1j * isqrt2, isqrt2]
        ], dtype=complex)

    def adjoint(self):
        return X2PGate(wires=self.wires)

class Y2PGate(Operation):
    r"""Native Y rotation by +pi/2 (Sqrt Y)."""
    num_wires = 1
    num_params = 0
    grad_method = None

    @staticmethod
    def compute_matrix() -> TensorLike:
        isqrt2 = 1 / np.sqrt(2)
        return qml.math.array([
            [isqrt2, -1 * isqrt2],
            [1 * isqrt2, isqrt2]
        ], dtype=complex)

    def adjoint(self):
        return Y2MGate(wires=self.wires)

class Y2MGate(Operation):
    r"""Native Y rotation by -pi/2 (Sqrt Y dag)."""
    num_wires = 1
    num_params = 0
    grad_method = None

    @staticmethod
    def compute_matrix() -> TensorLike:
        isqrt2 = 1 / np.sqrt(2)
        return qml.math.array([
            [isqrt2, 1 * isqrt2],
            [-1 * isqrt2, isqrt2]
        ], dtype=complex)

    def adjoint(self):
        return Y2PGate(wires=self.wires)


class XY2PGate(Operation):
    r"""
    Rotation in XY plane by +pi/2. 
    Axis defined by phi.
    """
    num_wires = 1
    num_params = 1
    ndim_params = (0,)
    grad_method = "A" # 支持自动微分

    def __init__(self, phi: TensorLike, wires: WiresLike, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi) -> TensorLike:
        if qml.math.get_interface(phi) == "tensorflow":
             phi = qml.math.cast_like(phi, 1j)
        
        c = qml.math.cos(phi)
        s = qml.math.sin(phi)
        
        one = qml.math.cast_like(qml.math.ones_like(phi), 1j)
        neg_i = -1j * one
        
        # -i * e^{-i*phi}
        term1 = neg_i * (c - 1j * s)
        # -i * e^{i*phi}
        term2 = neg_i * (c + 1j * s)
        
        isqrt2 = 1 / np.sqrt(2)
        mat = _stack_matrix([one, term1, term2, one])
        return mat * isqrt2

    def adjoint(self):
        return XY2MGate(self.data[0], wires=self.wires)

class XY2MGate(Operation):
    r"""
    Rotation in XY plane by -pi/2.
    """
    num_wires = 1
    num_params = 1
    ndim_params = (0,)
    grad_method = "A"

    def __init__(self, phi: TensorLike, wires: WiresLike, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi) -> TensorLike:
        if qml.math.get_interface(phi) == "tensorflow":
             phi = qml.math.cast_like(phi, 1j)

        c = qml.math.cos(phi)
        s = qml.math.sin(phi)
        
        one = qml.math.cast_like(qml.math.ones_like(phi), 1j)
        pos_i = 1j * one
        
        # +i * e^{-i*phi}
        term1 = pos_i * (c - 1j * s)
        # +i * e^{i*phi}
        term2 = pos_i * (c + 1j * s)
        
        isqrt2 = 1 / np.sqrt(2)
        mat = _stack_matrix([one, term1, term2, one])
        return mat * isqrt2

    def adjoint(self):
        return XY2PGate(self.data[0], wires=self.wires)