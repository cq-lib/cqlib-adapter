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

"""Tests for native gate classes in native_gates.py."""

import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.tape import QuantumScript

from cqlib_adapter.pennylane_ext.native_gates import (
    X2PGate,
    X2MGate,
    Y2PGate,
    Y2MGate,
    XY2PGate,
    XY2MGate,
)


class TestX2PGate:
    """Tests for X2PGate (sqrt(X) with +pi/2 rotation)."""

    def test_matrix_shape(self):
        """Test that the gate matrix has correct shape."""
        matrix = X2PGate.compute_matrix()
        assert matrix.shape == (2, 2)

    def test_matrix_is_unitary(self):
        """Test that the gate matrix is unitary (U @ U^dag = I)."""
        matrix = X2PGate.compute_matrix().astype(complex)
        dagger = np.conj(matrix.T)
        product = matrix @ dagger
        identity = np.eye(2)
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_matrix_matches_expected(self):
        """Test matrix elements match expected values."""
        isqrt2 = 1 / np.sqrt(2)
        expected = np.array([
            [isqrt2, -1j * isqrt2],
            [-1j * isqrt2, isqrt2]
        ], dtype=complex)
        matrix = X2PGate.compute_matrix()
        np.testing.assert_allclose(matrix, expected, atol=1e-10)

    def test_adjoint_returns_x2m(self):
        """Test that adjoint() returns X2MGate on same wire."""
        gate = X2PGate(wires=0)
        adjoint_gate = gate.adjoint()
        assert isinstance(adjoint_gate, X2MGate)
        assert adjoint_gate.wires == gate.wires

    def test_num_wires(self):
        """Test num_wires class attribute."""
        assert X2PGate.num_wires == 1

    def test_num_params(self):
        """Test num_params class attribute."""
        assert X2PGate.num_params == 0

    def test_operation_in_circuit(self):
        """Test X2PGate can be used in a PennyLane circuit."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            X2PGate(wires=0)
            return qml.probs(wires=0)

        result = circuit()
        assert result.shape == (2,)


class TestX2MGate:
    """Tests for X2MGate (sqrt(X) with -pi/2 rotation)."""

    def test_matrix_shape(self):
        """Test that the gate matrix has correct shape."""
        matrix = X2MGate.compute_matrix()
        assert matrix.shape == (2, 2)

    def test_matrix_is_unitary(self):
        """Test that the gate matrix is unitary."""
        matrix = X2MGate.compute_matrix().astype(complex)
        dagger = np.conj(matrix.T)
        product = matrix @ dagger
        identity = np.eye(2)
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_matrix_matches_expected(self):
        """Test matrix elements match expected values."""
        isqrt2 = 1 / np.sqrt(2)
        expected = np.array([
            [isqrt2, 1j * isqrt2],
            [1j * isqrt2, isqrt2]
        ], dtype=complex)
        matrix = X2MGate.compute_matrix()
        np.testing.assert_allclose(matrix, expected, atol=1e-10)

    def test_adjoint_returns_x2p(self):
        """Test that adjoint() returns X2PGate."""
        gate = X2MGate(wires=0)
        adjoint_gate = gate.adjoint()
        assert isinstance(adjoint_gate, X2PGate)
        assert adjoint_gate.wires == gate.wires

    # 【修复点 1】: 修改了错误的组合门断言，并拆分成如下两个正确的用例
    def test_x2p_then_x2p_equals_x(self):
        """Test X2P @ X2P should give PauliX (up to global phase)."""
        mat_x2p = X2PGate.compute_matrix()
        product = mat_x2p @ mat_x2p
        expected_x = np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_allclose(np.abs(product), np.abs(expected_x), atol=1e-10)

    def test_x2p_then_x2m_equals_identity(self):
        """Test X2P @ X2M should give Identity."""
        mat_x2p = X2PGate.compute_matrix()
        mat_x2m = X2MGate.compute_matrix()
        product = mat_x2m @ mat_x2p
        expected_i = np.eye(2, dtype=complex)
        np.testing.assert_allclose(np.abs(product), np.abs(expected_i), atol=1e-10)


class TestY2PGate:
    """Tests for Y2PGate (sqrt(Y) with +pi/2 rotation)."""

    def test_matrix_shape(self):
        """Test that the gate matrix has correct shape."""
        matrix = Y2PGate.compute_matrix()
        assert matrix.shape == (2, 2)

    def test_matrix_is_unitary(self):
        """Test that the gate matrix is unitary."""
        matrix = Y2PGate.compute_matrix().astype(complex)
        dagger = np.conj(matrix.T)
        product = matrix @ dagger
        identity = np.eye(2)
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_adjoint_returns_y2m(self):
        """Test that adjoint() returns Y2MGate."""
        gate = Y2PGate(wires=0)
        adjoint_gate = gate.adjoint()
        assert isinstance(adjoint_gate, Y2MGate)
        assert adjoint_gate.wires == gate.wires

    def test_operation_in_circuit(self):
        """Test Y2PGate can be used in a PennyLane circuit."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            Y2PGate(wires=0)
            return qml.probs(wires=0)

        result = circuit()
        assert result.shape == (2,)


class TestY2MGate:
    """Tests for Y2MGate (sqrt(Y) with -pi/2 rotation)."""

    def test_matrix_shape(self):
        """Test that the gate matrix has correct shape."""
        matrix = Y2MGate.compute_matrix()
        assert matrix.shape == (2, 2)

    def test_matrix_is_unitary(self):
        """Test that the gate matrix is unitary."""
        matrix = Y2MGate.compute_matrix().astype(complex)
        dagger = np.conj(matrix.T)
        product = matrix @ dagger
        identity = np.eye(2)
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_adjoint_returns_y2p(self):
        """Test that adjoint() returns Y2PGate."""
        gate = Y2MGate(wires=0)
        adjoint_gate = gate.adjoint()
        assert isinstance(adjoint_gate, Y2PGate)
        assert adjoint_gate.wires == gate.wires

    # 【修复点 2】: 修改了错误的组合门断言，同样拆分成两个正确的用例
    def test_y2p_then_y2p_equals_y(self):
        """Test Y2P @ Y2P should give PauliY (up to global phase)."""
        mat_y2p = Y2PGate.compute_matrix()
        product = mat_y2p @ mat_y2p
        expected_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        np.testing.assert_allclose(np.abs(product), np.abs(expected_y), atol=1e-10)

    def test_y2p_then_y2m_equals_identity(self):
        """Test Y2P @ Y2M should give Identity."""
        mat_y2p = Y2PGate.compute_matrix()
        mat_y2m = Y2MGate.compute_matrix()
        product = mat_y2m @ mat_y2p
        expected_i = np.eye(2, dtype=complex)
        np.testing.assert_allclose(np.abs(product), np.abs(expected_i), atol=1e-10)


class TestXY2PGate:
    """Tests for XY2PGate (parameterized XY-plane rotation by +pi/2)."""

    def test_matrix_shape_with_scalar_param(self):
        """Test matrix shape with a scalar phi parameter."""
        matrix = XY2PGate.compute_matrix(pnp.array(0.5))
        assert matrix.shape == (2, 2)

    def test_matrix_shape_with_float_param(self):
        """Test matrix shape with a float phi parameter."""
        matrix = XY2PGate.compute_matrix(0.5)
        assert matrix.shape == (2, 2)

    def test_matrix_is_unitary(self):
        """Test that the gate matrix is unitary for any phi."""
        phi = pnp.array(0.7)
        matrix = XY2PGate.compute_matrix(phi).astype(complex)
        dagger = np.conj(matrix.T)
        product = matrix @ dagger
        identity = np.eye(2)
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_adjoint_returns_xy2m(self):
        """Test that adjoint() returns XY2MGate."""
        gate = XY2PGate(0.5, wires=0)
        adjoint_gate = gate.adjoint()
        assert isinstance(adjoint_gate, XY2MGate)

    def test_adjoint_preserves_phi(self):
        """Test adjoint preserves the phi parameter."""
        phi = pnp.array(0.7)
        gate = XY2PGate(phi, wires=0)
        adjoint_gate = gate.adjoint()
        np.testing.assert_allclose(adjoint_gate.data[0], phi, atol=1e-10)

    def test_num_wires(self):
        """Test num_wires class attribute."""
        assert XY2PGate.num_wires == 1

    def test_num_params(self):
        """Test num_params class attribute."""
        assert XY2PGate.num_params == 1

    def test_ndim_params(self):
        """Test ndim_params class attribute."""
        assert XY2PGate.ndim_params == (0,)

    def test_operation_in_circuit(self):
        """Test XY2PGate can be used in a PennyLane circuit."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(phi):
            XY2PGate(phi, wires=0)
            return qml.probs(wires=0)

        result = circuit(0.5)
        assert result.shape == (2,)

    # 【修复点 3】: 移除了包裹在矩阵外的 np.abs()，保留了复数相位的对比
    def test_different_phi_produces_different_matrix(self):
        """Test that different phi values produce different matrices."""
        matrix1 = XY2PGate.compute_matrix(0.0)
        matrix2 = XY2PGate.compute_matrix(np.pi / 2)
        assert not np.allclose(matrix1, matrix2, atol=1e-10)


class TestXY2MGate:
    """Tests for XY2MGate (parameterized XY-plane rotation by -pi/2)."""

    def test_matrix_shape_with_scalar_param(self):
        """Test matrix shape with a scalar phi parameter."""
        matrix = XY2MGate.compute_matrix(pnp.array(0.5))
        assert matrix.shape == (2, 2)

    def test_matrix_shape_with_float_param(self):
        """Test matrix shape with a float phi parameter."""
        matrix = XY2MGate.compute_matrix(0.5)
        assert matrix.shape == (2, 2)

    def test_matrix_is_unitary(self):
        """Test that the gate matrix is unitary for any phi."""
        phi = pnp.array(0.7)
        matrix = XY2MGate.compute_matrix(phi).astype(complex)
        dagger = np.conj(matrix.T)
        product = matrix @ dagger
        identity = np.eye(2)
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_adjoint_returns_xy2p(self):
        """Test that adjoint() returns XY2PGate."""
        gate = XY2MGate(0.5, wires=0)
        adjoint_gate = gate.adjoint()
        assert isinstance(adjoint_gate, XY2PGate)

    def test_adjoint_preserves_phi(self):
        """Test adjoint preserves the phi parameter."""
        phi = pnp.array(0.7)
        gate = XY2MGate(phi, wires=0)
        adjoint_gate = gate.adjoint()
        np.testing.assert_allclose(adjoint_gate.data[0], phi, atol=1e-10)

    def test_operation_in_circuit(self):
        """Test XY2MGate can be used in a PennyLane circuit."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(phi):
            XY2MGate(phi, wires=0)
            return qml.probs(wires=0)

        result = circuit(0.5)
        assert result.shape == (2,)


class TestGateInteraction:
    """Tests for interactions between different gate types."""

    def test_x2p_adjoint_inverse_relationship(self):
        """Test X2P and X2M are adjoints of each other."""
        x2p = X2PGate(wires=0)
        x2m = X2MGate(wires=0)
        assert x2p.adjoint() is not x2p
        assert x2m.adjoint() is not x2m
        assert isinstance(x2p.adjoint(), X2MGate)
        assert isinstance(x2m.adjoint(), X2PGate)

    def test_y2p_adjoint_inverse_relationship(self):
        """Test Y2P and Y2M are adjoints of each other."""
        y2p = Y2PGate(wires=0)
        y2m = Y2MGate(wires=0)
        assert isinstance(y2p.adjoint(), Y2MGate)
        assert isinstance(y2m.adjoint(), Y2PGate)

    def test_xy2p_and_xy2m_are_adjoint_pairs(self):
        """Test XY2P and XY2M are adjoint pairs."""
        phi = 0.3
        gate_xy2p = XY2PGate(phi, wires=0)
        gate_xy2m = XY2MGate(phi, wires=0)
        assert isinstance(gate_xy2p.adjoint(), XY2MGate)
        assert isinstance(gate_xy2m.adjoint(), XY2PGate)

    def test_gates_work_on_multi_qubit_circuit(self):
        """Test gates can be used in multi-qubit circuits."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(phi):
            X2PGate(wires=0)
            Y2PGate(wires=1)
            XY2PGate(phi, wires=2)
            return qml.probs(wires=[0, 1, 2])

        result = circuit(0.5)
        assert result.shape == (8,)  # 2^3 = 8