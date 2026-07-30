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

"""Manual Qiskit check: Qiskit circuit -> cqlib circuit -> QCIS."""

from qiskit import QuantumCircuit

from cqlib_adapter.common import CompilationOptions
from cqlib_adapter.qiskit import compile_qiskit_circuit, qiskit_to_cqlib

NATIVE_BASIS = (
    "RZ",
    "X2P",
    "X2M",
    "Y2P",
    "Y2M",
    "XY2P",
    "XY2M",
    "CZ",
    "GPHASE",
)

circuit = QuantumCircuit(2, 2, name="bell")
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

bundle = qiskit_to_cqlib(circuit)
print("cqlib operations:")
for operation in bundle.circuit.operations:
    print(" ", operation)

artifact = compile_qiskit_circuit(
    circuit,
    options=CompilationOptions(target_basis=NATIVE_BASIS, seed=19),
)
print("\nQCIS sent to Tianyan:")
print(artifact.qcis)
print("measurement mapping:", artifact.measurements)
