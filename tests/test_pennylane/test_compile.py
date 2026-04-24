# test_gradients.py
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

"""Tests for compilation functions in compilation.py."""

import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.tape import QuantumScript

from cqlib_adapter.pennylane_ext.compilation import (
    compile_to_native_gates,
    compile_to_native_cqlib,
)
from cqlib_adapter.pennylane_ext.native_gates import (
    X2PGate,
    X2MGate,
    Y2PGate,
    Y2MGate,
    XY2PGate,
    XY2MGate,
)


class TestCompileToNativeGates:
    """Tests for compile_to_native_gates function."""

    def test_compile_hadamard(self):
        """Test Hadamard gate compilation to native gates."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.probs(wires=0)

        tape = compile_to_native_gates(circuit)
        op_names = [op.name for op in tape.operations]

        # H should decompose to RZ(pi/2) - X2P - RZ(pi/2)
        assert "RZ" in op_names
        assert "X2PGate" in op_names

    def test_compile_rx(self):
        """Test RX gate compilation to native gates."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RX(theta, wires=0)
            return qml.probs(wires=0)

        tape = compile_to_native_gates(circuit, pnp.pi / 4)
        op_names = [op.name for op in tape.operations]

        # RX should contain X2P, X2M, RZ gates
        assert "X2PGate" in op_names
        assert "X2MGate" in op_names

    def test_compile_ry(self):
        """Test RY gate compilation to native gates."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RY(theta, wires=0)
            return qml.probs(wires=0)

        tape = compile_to_native_gates(circuit, pnp.pi / 4)
        op_names = [op.name for op in tape.operations]

        assert "X2PGate" in op_names
        assert "X2MGate" in op_names

    def test_compile_paulix(self):
        """Test PauliX gate compilation."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        tape = compile_to_native_gates(circuit)
        op_names = [op.name for op in tape.operations]

        # X -> X2P - X2P
        assert op_names.count("X2PGate") == 2

    def test_compile_pauliy(self):
        """Test PauliY gate compilation."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliY(wires=0)
            return qml.probs(wires=0)

        tape = compile_to_native_gates(circuit)
        op_names = [op.name for op in tape.operations]

        # Y -> Y2P - Y2P
        assert op_names.count("Y2PGate") == 2

    def test_native_gates_unchanged(self):
        """Test that native gates pass through unchanged."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            X2PGate(wires=0)
            XY2PGate(0.5, wires=0)
            return qml.probs(wires=0)

        tape = compile_to_native_gates(circuit)
        op_names = [op.name for op in tape.operations]

        assert "X2PGate" in op_names
        assert "XY2PGate" in op_names

    def test_compile_with_measurements(self):
        """Test that measurements are preserved after compilation."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        tape = compile_to_native_gates(circuit)

        assert len(tape.measurements) == 1

    def test_compile_cnot(self):
        """Test CNOT gate compilation."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        tape = compile_to_native_gates(circuit)
        op_names = [op.name for op in tape.operations]

        # CNOT -> H-CZ-H on target
        assert "CZ" in op_names


class TestCompileToNativeCqlib:
    """Tests for compile_to_native_cqlib function."""

    def test_compile_hadamard_cqlib(self):
        """Test Hadamard compiles to CQLib native format."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.state()

        cql_circ = compile_to_native_cqlib(circuit)
        qcis = cql_circ.qcis

        # Should contain rz and y2p for H
        assert "rz" in qcis.lower() or "y2p" in qcis.lower()

    def test_compile_rx_cqlib(self):
        """Test RX compiles to CQLib native format."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RX(theta, wires=0)
            return qml.state()

        cql_circ = compile_to_native_cqlib(circuit, pnp.pi / 4)
        qcis = cql_circ.qcis.lower()

        # RX decomposition uses x2p, x2m, rz
        assert "x2p" in qcis or "x2m" in qcis or "rz" in qcis

    def test_compile_ry_cqlib(self):
        """Test RY compiles to CQLib native format."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RY(theta, wires=0)
            return qml.state()

        cql_circ = compile_to_native_cqlib(circuit, pnp.pi / 4)
        qcis = cql_circ.qcis.lower()

        assert "x2p" in qcis or "x2m" in qcis or "rz" in qcis

    def test_compile_paulix_cqlib(self):
        """Test PauliX compiles to CQLib native format."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.state()

        cql_circ = compile_to_native_cqlib(circuit)
        qcis = cql_circ.qcis.lower()

        # X -> x2p x2p
        assert qcis.count("x2p") == 2

    def test_compile_pauliy_cqlib(self):
        """Test PauliY compiles to CQLib native format."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliY(wires=0)
            return qml.state()

        cql_circ = compile_to_native_cqlib(circuit)
        qcis = cql_circ.qcis.lower()

        # Y -> y2p y2p
        assert qcis.count("y2p") == 2

    def test_compile_cnot_cqlib(self):
        """Test CNOT compiles to CQLib native format."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()

        cql_circ = compile_to_native_cqlib(circuit)
        qcis = cql_circ.qcis.lower()

        # CNOT decomposition uses y2m, cz, y2p
        assert "y2m" in qcis or "cz" in qcis or "y2p" in qcis

    def test_compile_s_gate_cqlib(self):
        """Test S gate compiles to CQLib native format."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.S(wires=0)
            return qml.state()

        cql_circ = compile_to_native_cqlib(circuit)
        qcis = cql_circ.qcis.lower()

        # S = RZ(pi/2)
        assert "rz" in qcis

    def test_compile_t_gate_cqlib(self):
        """Test T gate compiles to CQLib native format."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.T(wires=0)
            return qml.state()

        cql_circ = compile_to_native_cqlib(circuit)
        qcis = cql_circ.qcis.lower()

        # T = RZ(pi/4)
        assert "rz" in qcis
