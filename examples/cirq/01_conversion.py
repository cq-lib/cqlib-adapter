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

"""Manual Cirq check: Cirq Circuit -> cqlib Circuit -> native QCIS."""

from __future__ import annotations

import cirq

from cqlib_adapter.cirq import cirq_to_cqlib, compile_cirq_circuit
from cqlib_adapter.common import CompilationOptions

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


def main() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, key="control"),
        cirq.measure(q1, key="target"),
    )
    print("Cirq circuit:")
    print(circuit)

    bundle = cirq_to_cqlib(circuit)
    print("\ncqlib operations:")
    for operation in bundle.circuit.operations:
        print(" ", operation)
    print("measurement-key bits:", bundle.metadata.extras["measurement_key_bits"])

    artifact = compile_cirq_circuit(
        circuit, options=CompilationOptions(target_basis=NATIVE_BASIS, seed=31)
    )
    print("\nTianyan-compatible QCIS:")
    print(artifact.qcis)
    print("compiled measurement mapping:", artifact.measurements)
    print("PASS: Cirq -> cqlib -> QCIS conversion succeeded.")


if __name__ == "__main__":
    main()
