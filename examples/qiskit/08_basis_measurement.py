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

"""Verify explicit Qiskit X/Y-basis measurement through local cqlib QCIS."""

from qiskit import QuantumCircuit

from cqlib_adapter.qiskit import CqlibSimulatorBackend

SHOTS = 64


def main() -> None:
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)  # |+>, the +1 eigenstate of X.
    circuit.h(1)
    circuit.s(1)  # |+i>, the +1 eigenstate of Y.

    # Qiskit measures Z; rotate the requested X/Y bases into Z first.
    circuit.h(0)
    circuit.sdg(1)
    circuit.h(1)
    circuit.measure([0, 1], [0, 1])

    backend = CqlibSimulatorBackend(2)
    job = backend.run(circuit, shots=SHOTS, seed=29)
    counts = job.result().get_counts()
    print("Qiskit X/Y-basis counts:", counts)
    print("compiled QCIS:\n", job.qcis[0])
    if counts != {"00": SHOTS}:
        raise AssertionError("X/Y-basis rotations were not preserved by compilation")
    print("PASS: Qiskit basis rotations survived Qiskit -> cqlib -> QCIS simulation.")


if __name__ == "__main__":
    main()
