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


"""Quantum circuit compilation module for PennyLane and CQLib.

This module provides functions to transform standard PennyLane quantum circuits
into native gate sequences compatible with the CQLib hardware backend.
"""

from typing import Any, List, Union

import numpy as np
import pennylane as qml
from cqlib import Circuit
from pennylane.tape import QuantumScript
from pennylane.transforms import decompose
from pennylane.workflow import construct_tape

# Local application imports
from .native_gates import (
    X2MGate,
    X2PGate,
    XY2MGate,
    XY2PGate,
    Y2MGate,
    Y2PGate,
)

# Constants
PI = np.pi
PI_2 = np.pi / 2

NATIVE_GATES_TO_RETAIN = {
    qml.CNOT, qml.CZ, qml.Hadamard,
    qml.RX, qml.RY, qml.RZ,
    qml.PauliX, qml.PauliY, qml.PauliZ, 
    qml.S, qml.T,
    qml.Barrier, qml.Identity
}

def compile_to_native_gates(qnode: Any, *params: Any) -> QuantumScript:
    """Decomposes a QNode into a native gate set as a PennyLane QuantumScript.

    Args:
        qnode: The PennyLane QNode to be compiled.
        *params: Parameters required to execute the QNode.

    Returns:
        A QuantumScript containing the compiled native PennyLane gates.
    """
    # Construct the tape from the QNode with specific parameters
    tape = construct_tape(qnode)(*params)

    # Decompose into a standard base gate set first
    batch, _ = decompose(
        tape, 
        gate_set=NATIVE_GATES_TO_RETAIN, 
        max_expansion=10
    )
    decomposed_tape: QuantumScript = batch[0]

    new_ops = []

    for op in decomposed_tape.operations:
        name = op.name
        wires = op.wires
        p = op.parameters

        # Mapping logic
        if name == "Hadamard":
            # Map H -> RZ(pi/2) - X2P - RZ(pi/2)
            new_ops.append(qml.RZ(PI_2, wires=wires))
            new_ops.append(X2PGate(wires=wires))
            new_ops.append(qml.RZ(PI_2, wires=wires))

        elif name == "CNOT":
            # Map CNOT -> H(target) - CZ - H(target)
            ctrl, target = wires[0], wires[1]
            target_wires = [target]
            # H on target
            new_ops.append(qml.RZ(PI_2, wires=target_wires))
            new_ops.append(X2PGate(wires=target_wires))
            new_ops.append(qml.RZ(PI_2, wires=target_wires))
            # Native CZ
            new_ops.append(qml.CZ(wires=[ctrl, target]))
            # H on target again
            new_ops.append(qml.RZ(PI_2, wires=target_wires))
            new_ops.append(X2PGate(wires=target_wires))
            new_ops.append(qml.RZ(PI_2, wires=target_wires))

        elif name == "RX":
            # Map RX(theta) -> RZ(-pi/2) - X2P - RZ(theta) - X2M - RZ(pi/2)
            theta = p[0]
            new_ops.append(qml.RZ(-PI_2, wires=wires))
            new_ops.append(X2PGate(wires=wires))
            new_ops.append(qml.RZ(theta, wires=wires))
            new_ops.append(X2MGate(wires=wires))
            new_ops.append(qml.RZ(PI_2, wires=wires))

        elif name == "RY":
            # Map RY(theta) -> X2P - RZ(theta) - X2M
            theta = p[0]
            new_ops.append(X2PGate(wires=wires))
            new_ops.append(qml.RZ(theta, wires=wires))
            new_ops.append(X2MGate(wires=wires))

        elif name == "PauliX":
            # Map X -> X2P - X2P
            new_ops.append(X2PGate(wires=wires))
            new_ops.append(X2PGate(wires=wires))

        elif name == "PauliY":
            # Map Y -> Y2P - Y2P
            new_ops.append(Y2PGate(wires=wires))
            new_ops.append(Y2PGate(wires=wires))
            
        elif name == "PauliZ":
            new_ops.append(qml.RZ(PI, wires=wires))

        elif name == "S":
            new_ops.append(qml.RZ(PI_2, wires=wires))

        elif name == "T":
            new_ops.append(qml.RZ(PI / 4, wires=wires))

        elif name in ["RZ", "CZ", "Barrier", "Identity"]:
            # Retain standard gates that are native
            new_ops.append(op)

        elif name in [
            "X2PGate", "X2MGate", "Y2PGate", "Y2MGate", "XY2PGate", "XY2MGate"
        ]:
            # Retain custom native gates
            new_ops.append(op)

        else:
            print(f"Warning: Operation {name} not natively mapped. Kept.")
            new_ops.append(op)

    return QuantumScript(ops=new_ops, measurements=decomposed_tape.measurements)


def compile_to_native_cqlib(qnode: Any, *params: Any) -> Circuit:
    """Reconstructs a PennyLane circuit into a strict CQLib native circuit.

    Args:
        qnode: The PennyLane QNode to be compiled.
        *params: Parameters required to execute the QNode.

    Returns:
        A cqlib.Circuit object containing strictly native instructions.
    """
    # 1. Get decomposed Tape
    raw_tape = construct_tape(qnode)(*params)
    
    batch, _ = decompose(raw_tape, gate_set=NATIVE_GATES_TO_RETAIN, max_expansion=10)
    tape: QuantumScript = batch[0]

    # 2. Initialize CQLib circuit
    cql_circ = Circuit(tape.num_wires)

    # 3. Explicit manual mapping to physical native gates
    for op in tape.operations:
        name = op.name
        w = op.wires.tolist()
        p = op.parameters

        if name == "Hadamard":
            # H Q1 -> RZ Q1 PI, Y2P Q1
            cql_circ.rz(w[0], PI)
            cql_circ.y2p(w[0])

        elif name == "CNOT":
            # CX Q0 Q1 -> Y2M Q1, CZ Q0 Q1, Y2P Q1
            cql_circ.y2m(w[1])
            cql_circ.cz(w[0], w[1])
            cql_circ.y2p(w[1])

        elif name == "RX":
            # RX Q1 theta -> RZ Q1 PI_2, X2P Q1, RZ Q1 theta, X2M Q1, RZ Q1 -PI_2
            theta = p[0]
            cql_circ.rz(w[0], PI_2)
            cql_circ.x2p(w[0])
            cql_circ.rz(w[0], theta)
            cql_circ.x2m(w[0])
            cql_circ.rz(w[0], -PI_2)

        elif name == "RY":
            # RY Q1 theta -> X2P Q1, RZ Q1 theta, X2M Q1
            theta = p[0]
            cql_circ.x2p(w[0])
            cql_circ.rz(w[0], theta)
            cql_circ.x2m(w[0])

        elif name == "PauliX":
            # X Q1 -> X2P Q1, X2P Q1
            cql_circ.x2p(w[0])
            cql_circ.x2p(w[0])

        elif name == "PauliY":
            # Y Q1 -> Y2P Q1, Y2P Q1
            cql_circ.y2p(w[0])
            cql_circ.y2p(w[0])

        elif name == "PauliZ":
            cql_circ.rz(w[0], PI)

        elif name == "S":
            cql_circ.rz(w[0], PI_2)

        elif name == "T":
            cql_circ.rz(w[0], PI / 4)

        elif name == "RZ":
            cql_circ.rz(w[0], p[0])

        elif name == "CZ":
            cql_circ.cz(w[0], w[1])

        elif name == "Barrier":
            cql_circ.barrier(*w)

        elif name == "Identity":
            cql_circ.i(w[0], 0)

        # Handling custom native gates
        elif name == "X2PGate":
            cql_circ.x2p(w[0])
        elif name == "X2MGate":
            cql_circ.x2m(w[0])
        elif name == "Y2PGate":
            cql_circ.y2p(w[0])
        elif name == "Y2MGate":
            cql_circ.y2m(w[0])
        elif name == "XY2PGate":
            cql_circ.xy2p(w[0], p[0])
        elif name == "XY2MGate":
            cql_circ.xy2m(w[0], p[0])

        else:
            print(f"Warning: Operation {name} not mapped to native CQLib.")

    # 4. Final measurement processing
    if tape.measurements:
        cql_circ.measure_all()

    return cql_circ


if __name__ == "__main__":
    # Example execution
    
    @qml.qnode(qml.device("default.qubit", wires=3))
    def circuit() -> Any:
        """Example PennyLane QNode with standard gates."""
        qml.Hadamard(wires=0)
        qml.Toffoli(wires=[0, 1, 2])
        return qml.state()
        
    cqlib_circuit = compile_to_native_cqlib(circuit)
    print("Compiled QCIS Instructions:")
    print(cqlib_circuit.qcis)