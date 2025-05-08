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
Adapter module for integrating different interfaces.

This module provides classes and methods to adapt various interfaces,
making them compatible with the expected input/output format.
"""

from cqlib import Circuit as CqlibCircuit
from qiskit.circuit import QuantumCircuit as QiskitCircuit


def to_cqlib(qiskit_circuit: QiskitCircuit) -> CqlibCircuit:
    """
    Converts a Qiskit QuantumCircuit to a CqlibCircuit.

    Args:
        qiskit_circuit (QiskitCircuit): The Qiskit quantum circuit to be converted.

    Returns:
        CqlibCircuit: The converted circuit in the Cqlib format.

    Raises:
        NotImplementedError: If an operation in the Qiskit circuit is not supported
                             by the CqlibCircuit.
    """
    qubits_mapping = {q: i for i, q in enumerate(qiskit_circuit.qubits)}
    clbits_mapping = {c: i for i, c in enumerate(qiskit_circuit.clbits)}
    measure_qubits = {}
    cqlib_circuit = CqlibCircuit(list(qubits_mapping.values()))

    for ins in qiskit_circuit.data:
        operation = ins.operation
        qs = [qubits_mapping[q] for q in ins.qubits]
        cs = [clbits_mapping[c] for c in ins.clbits]

        if operation.name == "measure":
            measure_qubits[cs[0]] = qs[0]
        elif operation.name == "barrier":
            cqlib_circuit.barrier(*qs)
        elif operation.name == "i":
            cqlib_circuit.i(qs[0], 30)
        elif operation.name == 'unitary':
            ps = operation.params
            if len(ps) > 0:
                ps = ps[1:]
            getattr(cqlib_circuit, operation.label)(*qs, *ps)
        elif hasattr(cqlib_circuit, operation.name):
            getattr(cqlib_circuit, operation.name)(*qs, *operation.params)
        elif operation.name == 'tdg':
            cqlib_circuit.td(*qs, *operation.params)
        elif operation.name == 'sdg':
            cqlib_circuit.sd(*qs, *operation.params)
        elif operation.name == 'global_phase':
            pass
        else:
            raise NotImplementedError(f"{operation.name} is not supported")
    for cl_bit in sorted(measure_qubits.keys()):
        cqlib_circuit.measure(measure_qubits[cl_bit])

    return cqlib_circuit
