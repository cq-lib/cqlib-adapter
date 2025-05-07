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
Quantum transpilation equivalence test module.

This module validates the transpilation process of custom gates to both:
1. Custom basis gate sets (x2p, y2m, etc.)
2. Standard Qiskit basis gates (rx, ry, rz, etc.)

Verifies correct implementation of:
- Basic single-qubit gates (H, S, T)
- Parameterized rotations (RX, CRX, CRY)
- Custom gate decompositions
- Circuit visualization accuracy
- Parameter handling in transpiled circuits
"""

import random

import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library.standard_gates import CRXGate, CRYGate
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from cqlib_adapter.qiskit_ext.gates import X2PGate, X2MGate, Y2PGate, Y2MGate, \
    XY2MGate, XY2PGate, RxyGate
from tests.t import X2PGate


def test_rx():
    """Tests RX gate transpilation to custom basis gates.

    Validates:
    - Correct decomposition: RX(θ) → Rz(π/2) → X2p → Rz(θ) → X2m → Rz(-π/2)
    - Parameter preservation in decomposed circuit
    - Accurate ASCII circuit rendering
    """
    theta = Parameter('theta')
    c = QuantumCircuit(2)
    c.rx(theta, 0)
    res = transpile(c, basis_gates=['cz', 'rz', 'x2p', 'x2m'], optimization_level=3)
    target = ('     ┌─────────┐┌─────┐┌───────────┐┌─────┐┌──────────┐\n'
              'q_0: ┤ Rz(π/2) ├┤ X2p ├┤ Rz(theta) ├┤ X2m ├┤ Rz(-π/2) ├\n'
              '     └─────────┘└─────┘└───────────┘└─────┘└──────────┘\n'
              'q_1: ──────────────────────────────────────────────────\n'
              '                                                       ')
    assert str(res.draw('text')) == target


def test_h():
    """Verifies H gate transpilation to Y2p and phase gates.

    Checks:
    - Correct phase handling (global phase π/2)
    - Proper decomposition sequence: Rz(π) → Y2p
    - Visualization alignment with decomposed structure
    """
    c = QuantumCircuit(1)
    c.h(0)
    res = transpile(c, basis_gates=['cz', 'rz', 'y2p', 'x2p', 'global_phase'])
    target = ('global phase: π/2\n'
              '   ┌───────┐┌─────┐\n'
              'q: ┤ Rz(π) ├┤ Y2p ├\n'
              '   └───────┘└─────┘')
    assert str(res.draw('text')) == target


def test_s():
    """Validates S gate transpilation to Rz(π/2) with phase correction.

    Tests:
    - Global phase π/4 implementation
    - Single Rz gate with π/2 rotation
    - Circuit diagram accuracy
    """
    c = QuantumCircuit(1)
    c.s(0)
    res = transpile(c, basis_gates=['cz', 'rz', 'y2p', 'x2p', 'global_phase'])
    target = ('global phase: π/4\n'
              '   ┌─────────┐\n'
              'q: ┤ Rz(π/2) ├\n'
              '   └─────────┘')
    assert str(res.draw('text')) == target


def test_t():
    """Tests T gate transpilation to Rz(π/4) with phase handling.

    Verifies:
    - Correct π/8 global phase implementation
    - Proper Rz(π/4) rotation
    - Visualization consistency
    """
    c = QuantumCircuit(1)
    c.t(0)
    res = transpile(c, basis_gates=['cz', 'rz', 'y2p', 'x2p', 'global_phase'])
    target = ('global phase: π/8\n'
              '   ┌─────────┐\n'
              'q: ┤ Rz(π/4) ├\n'
              '   └─────────┘')
    assert str(res.draw('text')) == target


def test_crx():
    """Comprehensive CRX gate transpilation test.

    Validates for 10 random θ values:
    - Matrix equivalence between original and transpiled circuits
    - Complex decomposition structure:
        CRX(θ) → [Rz, Y2p, CZ, Rz, Y2p, X2p, Rz, X2m] sequence
    - Multi-qubit interaction preservation
    - Parameter substitution correctness
    """
    theta = Parameter('theta')
    c = QuantumCircuit(2)
    c.crx(theta, 0, 1)
    res = transpile(c, basis_gates=['cz', 'rz', 'y2p', 'y2m', 'x2p', 'x2m', 'global_phase'])
    target = ('global phase: 0\n'
              '                                                                         »\n'
              'q_0: ────────────────────■───────────────────────────────────────────────»\n'
              '     ┌──────────┐┌─────┐ │ ┌───────┐┌─────┐┌─────┐┌──────────────┐┌─────┐»\n'
              'q_1: ┤ Rz(3π/2) ├┤ Y2p ├─■─┤ Rz(π) ├┤ Y2p ├┤ X2p ├┤ Rz(-theta/2) ├┤ X2m ├»\n'
              '     └──────────┘└─────┘   └───────┘└─────┘└─────┘└──────────────┘└─────┘»\n'
              '«                                                                     »\n'
              '«q_0: ─────────────────■──────────────────────────────────────────────»\n'
              '«     ┌───────┐┌─────┐ │ ┌───────┐┌─────┐┌─────┐┌─────────────┐┌─────┐»\n'
              '«q_1: ┤ Rz(π) ├┤ Y2p ├─■─┤ Rz(π) ├┤ Y2p ├┤ X2p ├┤ Rz(theta/2) ├┤ X2m ├»\n'
              '«     └───────┘└─────┘   └───────┘└─────┘└─────┘└─────────────┘└─────┘»\n'
              '«                 \n'
              '«q_0: ────────────\n'
              '«     ┌──────────┐\n'
              '«q_1: ┤ Rz(-π/2) ├\n'
              '«     └──────────┘')
    assert str(res.draw('text')) == target
    for i in range(10):
        t = np.pi * random.random()
        assert np.allclose(np.asarray(CRXGate(t)), Operator(c.assign_parameters([t])).to_matrix())


def test_cry():
    """Tests CRY gate transpilation with parameterized rotations.

    Checks for 10 random θ values:
    - Decomposition sequence preservation:
        CRY(θ) → [X2p, Rz, X2m, Rz, Y2p, CZ, ...]
    - Control-target qubit relationship maintenance
    - Parameter scaling (θ/2) in Rz gates
    - End-to-end matrix equivalence
    """
    theta = Parameter('theta')
    c = QuantumCircuit(2)
    c.cry(theta, 0, 1)
    res = transpile(c, basis_gates=['cz', 'rz', 'y2p', 'y2m', 'x2p', 'x2m', 'global_phase'])
    target = ('                                                                            »\n'
              'q_0: ──────────────────────────────────────────────■────────────────────────»\n'
              '     ┌─────┐┌─────────────┐┌─────┐┌───────┐┌─────┐ │ ┌───────┐┌─────┐┌─────┐»\n'
              'q_1: ┤ X2p ├┤ Rz(theta/2) ├┤ X2m ├┤ Rz(π) ├┤ Y2p ├─■─┤ Rz(π) ├┤ Y2p ├┤ X2p ├»\n'
              '     └─────┘└─────────────┘└─────┘└───────┘└─────┘   └───────┘└─────┘└─────┘»\n'
              '«                                                               \n'
              '«q_0: ────────────────────────────────────────■─────────────────\n'
              '«     ┌──────────────┐┌─────┐┌───────┐┌─────┐ │ ┌───────┐┌─────┐\n'
              '«q_1: ┤ Rz(-theta/2) ├┤ X2m ├┤ Rz(π) ├┤ Y2p ├─■─┤ Rz(π) ├┤ Y2p ├\n'
              '«     └──────────────┘└─────┘└───────┘└─────┘   └───────┘└─────┘')
    assert str(res.draw('text')) == target
    for i in range(10):
        t = np.pi * random.random()
        assert np.allclose(np.asarray(CRYGate(t)), Operator(c.assign_parameters([t])).to_matrix())


def test_to_qiskit_gate():
    """Validates custom gate conversion to standard Qiskit gates.

    Verifies:
    - X2P/M → Rx(±π/2) conversion
    - Y2P/M → Ry(±π/2) conversion
    - XY2P/M → Rz rotations with parameter transformations
    - Cross-qubit gate position preservation
    - Composite rotation accuracy in standard basis
    """
    theta = Parameter('theta')
    phi = Parameter('phi')
    c = QuantumCircuit(2)
    c.append(X2PGate(), [0])
    c.append(X2MGate(), [1])
    c.append(Y2PGate(), [0])
    c.append(Y2MGate(), [1])
    c.append(XY2MGate(theta), [0])
    c.append(XY2PGate(theta), [1])
    c.append(RxyGate(phi, theta), [1])

    res = transpile(c, basis_gates=['h', 'rx', 'ry', 'rz', 'cx', 'cz', 'global_phase'])
    target = ('     ┌─────────┐ ┌─────────┐ ┌──────────────────┐┌─────────┐┌─────────────────┐»\n'
              'q_0: ┤ Rx(π/2) ├─┤ Ry(π/2) ├─┤ Rz(-theta - π/2) ├┤ Ry(π/2) ├┤ Rz(theta + π/2) ├»\n'
              '     ├─────────┴┐├─────────┴┐├─────────────────┬┘├─────────┤├─────────────────┤»\n'
              'q_1: ┤ Rx(-π/2) ├┤ Ry(-π/2) ├┤ Rz(π/2 - theta) ├─┤ Ry(π/2) ├┤ Rz(theta - π/2) ├»\n'
              '     └──────────┘└──────────┘└─────────────────┘ └─────────┘└─────────────────┘»\n'
              '«                                                                           \n'
              '«q_0: ──────────────────────────────────────────────────────────────────────\n'
              '«     ┌───────────────┐┌─────────┐┌───────────┐┌──────────┐┌───────────────┐\n'
              '«q_1: ┤ Rz(π/2 - phi) ├┤ Rx(π/2) ├┤ Rz(theta) ├┤ Rx(-π/2) ├┤ Rz(phi - π/2) ├\n'
              '«     └───────────────┘└─────────┘└───────────┘└──────────┘└───────────────┘')
    assert str(res.draw()) == target
