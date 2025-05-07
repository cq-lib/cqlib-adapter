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
Test module for quantum gate equivalence validations.

This module contains comprehensive tests to verify the mathematical equivalence
between custom quantum gates and their Qiskit standard library counterparts.
Validates gate implementations including rotation gates (X2P, X2M, Y2P, Y2M),
composite gates (XY2P, XY2M), and parameterized gates (RxyGate).

Tests cover:
- Matrix equivalence validation
- Parameterized rotation behavior
- Circuit composition correctness
- Gate decomposition rules
- Visualization consistency
"""

from math import pi
from random import random

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import GlobalPhaseGate, RXGate, RYGate, \
    HGate, XGate, YGate
from qiskit.quantum_info import Operator

from cqlib_adapter.qiskit_ext.gates import X2PGate, X2MGate, Y2PGate, Y2MGate, \
    XY2PGate, XY2MGate, RxyGate


def test_apply_gate():
    """Verifies correct gate application and circuit visualization.

    Tests the proper rendering of custom gates in quantum circuits by:
    1. Creating a circuit with parameterized gates on multiple qubits
    2. Comparing the text drawing output with expected ASCII art

    Validates the correct ordering and positioning of:
    - Basic rotation gates (X2P, X2M, Y2P, Y2M)
    - Parameterized composite gates (XY2P, XY2M)
    - Multi-parameter gates (RxyGate)
    """
    phi = Parameter('phi')
    theta = Parameter('theta')
    c = QuantumCircuit(2)
    c.append(X2PGate(), [0])
    c.append(X2MGate(), [1])
    c.append(Y2PGate(), [0])
    c.append(Y2MGate(), [1])
    c.append(XY2PGate(theta), [0])
    c.append(XY2MGate(theta), [1])
    c.append(RxyGate(phi, theta), [1])
    text = ('     ┌─────┐┌─────┐┌─────────────┐                  \n'
            'q_0: ┤ X2p ├┤ Y2p ├┤ Xy2p(theta) ├──────────────────\n'
            '     ├─────┤├─────┤├─────────────┤┌────────────────┐\n'
            'q_1: ┤ X2m ├┤ Y2m ├┤ Xy2m(theta) ├┤ Rxy(phi,theta) ├\n'
            '     └─────┘└─────┘└─────────────┘└────────────────┘')
    assert str(c.draw('text')) == text


class TestGate:
    """Main test class containing gate equivalence validations."""

    def test_x2p(self):
        """Validates X2P gate equivalence with RX(π/2).

        Test cases:
        1. Direct equivalence between X2P and RX(π/2)
        2. Reverse equivalence verification
        """
        rx_qc = QuantumCircuit(1)
        rx_qc.append(X2PGate(), [0])
        assert np.allclose(np.asarray(RXGate(pi / 2)), Operator(rx_qc).to_matrix())

        x_qc = QuantumCircuit(1)
        x_qc.rx(pi / 2, 0)
        assert np.allclose(np.asarray(X2PGate()), Operator(x_qc).to_matrix())

    def test_x(self):
        """Verifies X gate decomposition using two X2P gates with phase correction."""
        x_qc = QuantumCircuit(1)
        x_qc.append(X2PGate(), [0])
        x_qc.append(X2PGate(), [0])
        x_qc.append(GlobalPhaseGate(pi / 2), [])
        assert np.allclose(np.asarray(XGate()), Operator(x_qc).to_matrix())

    def test_rx(self):
        """Tests parameterized RX gate implementation through composite rotations.

        Validates for 5 random θ values in [0, π] range:
        - RX(θ) ≡ Rz(π/2) → X2P → Rz(θ) → X2M → Rz(-π/2)
        """
        theta = Parameter("theta")
        rx_qc = QuantumCircuit(1)
        rx_qc.rz(pi / 2, 0)
        rx_qc.append(X2PGate(), [0])
        rx_qc.rz(theta, 0)
        rx_qc.append(X2MGate(), [0])
        rx_qc.rz(-pi / 2, 0)

        for i in range(5):
            t = pi * i / 5 * random()
            assert np.allclose(np.asarray(RXGate(t)), Operator(rx_qc.assign_parameters([t])).to_matrix())

    def test_ry(self):
        """Verifies RY gate implementation via X2P-based decomposition.

        Checks for 5 random θ values:
        - RY(θ) ≡ X2P → Rz(θ) → X2M
        """
        theta = Parameter("theta")
        ry_qc = QuantumCircuit(1)
        ry_qc.append(X2PGate(), [0])
        ry_qc.rz(theta, 0)
        ry_qc.append(X2MGate(), [0])

        for i in range(5):
            t = pi * i / 5 * random()
            assert np.allclose(np.asarray(RYGate(t)), Operator(ry_qc.assign_parameters([t])).to_matrix())

    def test_x2m(self):
        """Validates X2M gate equivalence with RX(-π/2)."""
        x2m_qc = QuantumCircuit(1)
        x2m_qc.append(X2MGate(), [0])
        assert np.allclose(np.asarray(RXGate(-pi / 2)), Operator(x2m_qc).to_matrix())

        x2m_qc = QuantumCircuit(1)
        x2m_qc.rx(-pi / 2, 0)
        assert np.allclose(np.asarray(X2MGate()), Operator(x2m_qc).to_matrix())

    def test_y2p(self):
        """Tests Y2P gate equivalence with RY(π/2)."""
        y2p_qc = QuantumCircuit(1)
        y2p_qc.append(Y2PGate(), [0])
        assert np.allclose(np.asarray(RYGate(pi / 2)), Operator(y2p_qc).to_matrix())

        ry_qc = QuantumCircuit(1)
        ry_qc.ry(pi / 2, 0)
        assert np.allclose(np.asarray(Y2PGate()), Operator(ry_qc).to_matrix())

    def test_y(self):
        """Verifies Y gate decomposition using two Y2P gates with phase adjustment."""
        y_qc = QuantumCircuit(1)
        y_qc.append(Y2PGate(), [0])
        y_qc.append(Y2PGate(), [0])
        y_qc.append(GlobalPhaseGate(pi / 2), [])
        assert np.allclose(np.asarray(YGate()), Operator(y_qc).to_matrix())

    def test_y2m(self):
        """Validates Y2M gate equivalence with RY(-π/2)."""
        y2m_qc = QuantumCircuit(1)
        y2m_qc.append(Y2MGate(), [0])
        assert np.allclose(np.asarray(RYGate(-pi / 2)), Operator(y2m_qc).to_matrix())

        ry_qc = QuantumCircuit(1)
        ry_qc.ry(-pi / 2, 0)
        assert np.allclose(np.asarray(Y2MGate()), Operator(ry_qc).to_matrix())

    def test_h(self):
        """Tests Hadamard gate decomposition using Y2P and phase rotations."""
        h_decomp = QuantumCircuit(1)
        h_decomp.rz(pi, 0)
        h_decomp.append(Y2PGate(), [0])
        h_decomp.append(GlobalPhaseGate(pi / 2), [])
        assert np.allclose(np.asarray(HGate()), Operator(h_decomp).to_matrix())

    def test_xy2p(self):
        """Validates XY2P gate implementation with 10 random θ values.

        Checks both direct matrix equivalence and circuit equivalence through:
        - XY2P(θ) ≡ Rz(π/2-θ) → RY(π/2) → Rz(θ-π/2)
        """
        theta = Parameter("theta")
        xy2p_qc = QuantumCircuit(1)
        xy2p_qc.rz(pi / 2 - theta, 0)
        xy2p_qc.ry(pi / 2, 0)
        xy2p_qc.rz(theta - pi / 2, 0)

        qc = QuantumCircuit(1)
        qc.append(XY2PGate(theta), [0])
        for i in range(10):
            t = pi * random()
            assert np.allclose(np.asarray(XY2PGate(t)), Operator(xy2p_qc.assign_parameters([t])).to_matrix())
            assert Operator(xy2p_qc.assign_parameters([t])).equiv(Operator(qc.assign_parameters([t])))

    def test_xy2m(self):
        """Tests XY2M gate with 5 random θ values.

        Verifies:
        - XY2M(θ) ≡ Rz(-π/2-θ) → RY(π/2) → Rz(θ+π/2)
        - Circuit equivalence through different construction methods
        """
        theta = Parameter("theta")
        xy2m_qc = QuantumCircuit(1)
        xy2m_qc.rz(-pi / 2 - theta, 0)
        xy2m_qc.ry(pi / 2, 0)
        xy2m_qc.rz(theta + pi / 2, 0)

        qc = QuantumCircuit(1)
        qc.append(XY2MGate(theta), [0])

        for i in range(5):
            t = pi * random()
            assert np.allclose(np.asarray(XY2MGate(t)), Operator(xy2m_qc.assign_parameters([t])).to_matrix())
            assert Operator(xy2m_qc.assign_parameters([t])).equiv(Operator(qc.assign_parameters([t])))

    def test_rxy(self):
        """Comprehensive RxyGate validation with randomized parameters.

        Tests for 5 random (φ, θ) pairs:
        1. Direct matrix equivalence
        2. Circuit composition equivalence
        3. Parameter substitution correctness
        """
        phi = Parameter("phi")
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rz(pi / 2 - phi, 0)
        qc.rx(pi / 2, 0)
        qc.rz(theta, 0)
        qc.rx(-pi / 2, 0)
        qc.rz(phi - pi / 2, 0)

        qc2 = QuantumCircuit(1)
        qc2.append(RxyGate(phi, theta), [0])

        for i in range(5):
            p = pi * random()
            t = -pi * random()
            assert np.allclose(
                np.asarray(RxyGate(p, t)),
                Operator(qc.assign_parameters({phi: p, theta: t})).to_matrix()
            )
            assert Operator(qc.assign_parameters({phi: p, theta: t})).equiv(
                Operator(qc2.assign_parameters({phi: p, theta: t}))
            )
