# This code is part of cqlib.
#
# Copyright (C) 2025 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Gates module for defining custom quantum gates and their equivalence rules.

This module provides custom quantum gates (e.g., X2P, X2M, Y2P, Y2M, XY2P, XY2M)
and their equivalence rules with standard Qiskit gates. It also includes
validation to ensure the correctness of the gate definitions.
"""

from math import pi

import numpy as np
from qiskit.quantum_info import Operator
from qiskit.circuit import Gate, QuantumCircuit, Parameter
from qiskit.circuit.library import GlobalPhaseGate, RXGate, RYGate, \
    HGate, XGate, YGate
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as SELib

from cqlib.circuits.gates.x import X2P, X2M
from cqlib.circuits.gates.y import Y2P, Y2M
from cqlib.circuits.gates.xy import XY2P, XY2M


class X2PGate(Gate):
    """
    Custom quantum gate representing a positive X rotation by π/2.

    This gate is equivalent to a rotation around the X-axis by π/2 radians.
    """

    def __init__(self, label=None):
        """
        Initializes the X2PGate.

        Args:
            label (str, optional): A custom label for the gate. Defaults to None.
        """
        super().__init__("x2p", 1, params=[], label=label)

    def _define(self):
        """Defines the quantum circuit for the X2P gate."""
        defn = QuantumCircuit(1)
        defn.rx(pi / 2, 0)
        self._definition = defn

    def __array__(self, dtype=None, copy=None):
        """
        Returns the matrix representation of the gate.

        Args:
            dtype: The data type of the array.
            copy: Whether to avoid copying the array.

        Returns:
            np.ndarray: The matrix representation of the gate.

        Raises:
            ValueError: If copying cannot be avoided.
        """
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.asarray(X2P(), dtype=dtype)


class X2MGate(Gate):
    """
    Custom quantum gate representing a negative X rotation by π/2.

    This gate is equivalent to a rotation around the X-axis by -π/2 radians.
    """

    def __init__(self, label=None):
        """
        Initializes the X2MGate.

        Args:
            label (str, optional): A custom label for the gate. Defaults to None.
        """
        super().__init__("x2m", 1, [], label=label)

    def _define(self):
        """Defines the quantum circuit for the X2M gate."""
        defn = QuantumCircuit(1)
        defn.rx(-pi / 2, 0)
        self._definition = defn

    def __array__(self, dtype=None, copy=None):
        """
        Returns the matrix representation of the gate.

        Args:
            dtype: The data type of the array.
            copy: Whether to avoid copying the array.

        Returns:
            np.ndarray: The matrix representation of the gate.

        Raises:
            ValueError: If copying cannot be avoided.
        """
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.asarray(X2M(), dtype=dtype)


class Y2PGate(Gate):
    """
    Custom quantum gate representing a positive Y rotation by π/2.

    This gate is equivalent to a rotation around the Y-axis by π/2 radians.
    """

    def __init__(self, label=None):
        """
        Initializes the Y2PGate.

        Args:
            label (str, optional): A custom label for the gate. Defaults to None.
        """
        super().__init__("y2p", 1, params=[], label=label)

    def _define(self):
        """Defines the quantum circuit for the Y2P gate."""
        defn = QuantumCircuit(1)
        defn.ry(pi / 2, 0)
        self._definition = defn

    def __array__(self, dtype=None, copy=None):
        """
        Returns the matrix representation of the gate.

        Args:
            dtype: The data type of the array.
            copy: Whether to avoid copying the array.

        Returns:
            np.ndarray: The matrix representation of the gate.

        Raises:
            ValueError: If copying cannot be avoided.
        """
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.asarray(Y2P(), dtype=dtype)


class Y2MGate(Gate):
    """
    Custom quantum gate representing a negative Y rotation by π/2.

    This gate is equivalent to a rotation around the Y-axis by -π/2 radians.
    """

    def __init__(self, label=None):
        """
        Initializes the Y2MGate.

        Args:
            label (str, optional): A custom label for the gate. Defaults to None.
        """
        super().__init__("y2m", 1, [], label=label)

    def _define(self):
        """Defines the quantum circuit for the Y2M gate."""
        defn = QuantumCircuit(1)
        defn.ry(-pi / 2, 0)
        self._definition = defn

    def __array__(self, dtype=None, copy=None):
        """
        Returns the matrix representation of the gate.

        Args:
            dtype: The data type of the array.
            copy: Whether to avoid copying the array.

        Returns:
            np.ndarray: The matrix representation of the gate.

        Raises:
            ValueError: If copying cannot be avoided.
        """
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.asarray(Y2M(), dtype=dtype)


class XY2PGate(Gate):
    """
    Custom quantum gate representing a positive XY rotation.

    This gate is parameterized by an angle `theta` and represents a rotation
    in the XY plane.
    """

    def __init__(self, theta: float | Parameter, label: str = None):
        """
        Initializes the XY2PGate.

        Args:
            theta (float|Parameter): The rotation angle.
            label (str, optional): A custom label for the gate. Defaults to None.
        """
        super().__init__("xy2p", 1, [theta], label=label)

    def _define(self):
        """Defines the quantum circuit for the XY2P gate."""
        theta_ = self.params[0]
        defn = QuantumCircuit(1)
        defn.rz(pi / 2 - theta_, 0)
        defn.ry(pi / 2, 0)
        defn.rz(theta_ - pi / 2, 0)
        self._definition = defn

    def __array__(self, dtype=None, copy=None):
        """
        Returns the matrix representation of the gate.

        Args:
            dtype: The data type of the array.
            copy: Whether to avoid copying the array.

        Returns:
            np.ndarray: The matrix representation of the gate.

        Raises:
            ValueError: If copying cannot be avoided.
        """
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.asarray(XY2P(self.params[0]), dtype=dtype)


class XY2MGate(Gate):
    """
    Custom quantum gate representing a negative XY rotation.

    This gate is parameterized by an angle `theta` and represents a rotation
    in the XY plane.
    """

    def __init__(self, theta: float | Parameter, label: str = None):
        """
        Initializes the XY2MGate.

        Args:
            theta (float | Parameter): The rotation angle.
            label (str, optional): A custom label for the gate. Defaults to None.
        """
        super().__init__("xy2m", 1, [theta], label=label)

    def _define(self):
        """Defines the quantum circuit for the XY2M gate."""
        theta_ = self.params[0]
        defn = QuantumCircuit(1)
        defn.rz(-pi / 2 - theta_, 0)
        defn.ry(pi / 2, 0)
        defn.rz(theta_ + pi / 2, 0)
        self._definition = defn

    def __array__(self, dtype=None, copy=None):
        """
        Returns the matrix representation of the gate.

        Args:
            dtype: The data type of the array.
            copy: Whether to avoid copying the array.

        Returns:
            np.ndarray: The matrix representation of the gate.

        Raises:
            ValueError: If copying cannot be avoided.
        """
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.asarray(XY2M(self.params[0]), dtype=dtype)


# Equivalence rules and validation
# rx pi/2
rx_qc = QuantumCircuit(1)
rx_qc.append(X2PGate(), [0])
# SELib.add_equivalence(RXGate(pi / 2), rx_qc)
assert np.allclose(np.asarray(RXGate(pi / 2)), Operator(rx_qc).to_matrix())

x_qc = QuantumCircuit(1)
x_qc.rx(pi / 2, 0)
SELib.add_equivalence(X2PGate(), x_qc)
assert np.allclose(np.asarray(X2PGate()), Operator(x_qc).to_matrix())

# x
x_qc = QuantumCircuit(1)
x_qc.append(X2PGate(), [0])
x_qc.append(X2PGate(), [0])
x_qc.append(GlobalPhaseGate(pi / 2), [])
SELib.add_equivalence(XGate(), x_qc)
assert np.allclose(np.asarray(XGate()), Operator(x_qc).to_matrix())

# RX
theta = Parameter("theta")
rx_qc = QuantumCircuit(1)
rx_qc.rz(pi / 2, 0)
rx_qc.append(X2PGate(), [0])
rx_qc.rz(theta, 0)
rx_qc.append(X2MGate(), [0])
rx_qc.rz(-pi / 2, 0)
SELib.add_equivalence(RXGate(theta), rx_qc)

for i in range(5):
    t = pi * i / 5
    assert np.allclose(np.asarray(RXGate(t)), Operator(rx_qc.assign_parameters([t])).to_matrix())

# RY
theta = Parameter("theta")
ry_qc = QuantumCircuit(1)
ry_qc.append(X2PGate(), [0])
ry_qc.rz(theta, 0)
ry_qc.append(X2MGate(), [0])
SELib.add_equivalence(RYGate(theta), ry_qc)

for i in range(5):
    t = pi * i / 5
    assert np.allclose(np.asarray(RYGate(t)), Operator(ry_qc.assign_parameters([t])).to_matrix())

# X2M
x2m_qc = QuantumCircuit(1)
x2m_qc.append(X2MGate(), [0])
# SELib.add_equivalence(RXGate(-pi / 2), x2m_qc)
assert np.allclose(np.asarray(RXGate(-pi / 2)), Operator(x2m_qc).to_matrix())

# X2M
x2m_qc = QuantumCircuit(1)
x2m_qc.rx(-pi / 2, 0)
SELib.add_equivalence(X2MGate(), x2m_qc)
assert np.allclose(np.asarray(X2MGate()), Operator(x2m_qc).to_matrix())

# Y2P
y2p_qc = QuantumCircuit(1)
y2p_qc.append(Y2PGate(), [0])
# SELib.add_equivalence(RYGate(pi / 2), y2p_qc)
assert np.allclose(np.asarray(RYGate(pi / 2)), Operator(y2p_qc).to_matrix())

ry_qc = QuantumCircuit(1)
ry_qc.ry(pi / 2, 0)
SELib.add_equivalence(Y2PGate(), ry_qc)
assert np.allclose(np.asarray(Y2PGate()), Operator(ry_qc).to_matrix())

# Y
y_qc = QuantumCircuit(1)
y_qc.append(Y2PGate(), [0])
y_qc.append(Y2PGate(), [0])
y_qc.append(GlobalPhaseGate(pi / 2), [])
SELib.add_equivalence(YGate(), y_qc)
assert np.allclose(np.asarray(YGate()), Operator(y_qc).to_matrix())

# X2M
y2m_qc = QuantumCircuit(1)
y2m_qc.append(Y2MGate(), [0])
# SELib.add_equivalence(RYGate(-pi / 2), y2m_qc)
assert np.allclose(np.asarray(RYGate(-pi / 2)), Operator(y2m_qc).to_matrix())

ry_qc = QuantumCircuit(1)
ry_qc.ry(-pi / 2, 0)
SELib.add_equivalence(Y2MGate(), ry_qc)
assert np.allclose(np.asarray(Y2MGate()), Operator(ry_qc).to_matrix())

#  H gate
h_decomp = QuantumCircuit(1)
h_decomp.rz(pi, 0)
h_decomp.append(Y2PGate(), [0])
h_decomp.append(GlobalPhaseGate(pi / 2), [])
SELib.add_equivalence(HGate(), h_decomp)
assert np.allclose(np.asarray(HGate()), Operator(h_decomp).to_matrix())

# XY2P
theta = Parameter("theta")
xy2p_qc = QuantumCircuit(1)
xy2p_qc.rz(pi / 2 - theta, 0)
xy2p_qc.ry(pi / 2, 0)
xy2p_qc.rz(theta - pi / 2, 0)
SELib.add_equivalence(XY2PGate(theta), xy2p_qc)

qc = QuantumCircuit(1)
qc.append(XY2PGate(theta), [0])
for i in range(5):
    t = pi * i / 5
    assert np.allclose(np.asarray(XY2PGate(t)), Operator(xy2p_qc.assign_parameters([t])).to_matrix())
    assert Operator(xy2p_qc.assign_parameters([t])).equiv(Operator(qc.assign_parameters([t])))

# XY2M
theta = Parameter("theta")
xy2m_qc = QuantumCircuit(1)
xy2m_qc.rz(-pi / 2 - theta, 0)
xy2m_qc.ry(pi / 2, 0)
xy2m_qc.rz(theta + pi / 2, 0)
SELib.add_equivalence(XY2MGate(theta), xy2m_qc)

qc = QuantumCircuit(1)
qc.append(XY2MGate(theta), [0])

for i in range(5):
    t = pi * i / 5
    assert np.allclose(np.asarray(XY2MGate(t)), Operator(xy2m_qc.assign_parameters([t])).to_matrix())
    assert Operator(xy2m_qc.assign_parameters([t])).equiv(Operator(qc.assign_parameters([t])))

__all__ = [
    'X2PGate',
    'X2MGate',
    'Y2PGate',
    'Y2MGate',
    'XY2PGate',
    'XY2MGate'
]
