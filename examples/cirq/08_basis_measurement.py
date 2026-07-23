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

"""Verify explicit Cirq X/Y-basis measurement through local cqlib QCIS."""

import cirq

from cqlib_adapter.cirq import CqlibSimulatorSampler

REPETITIONS = 64


def main() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),  # |+>, the +1 eigenstate of X.
        cirq.H(q1),
        cirq.S(q1),  # |+i>, the +1 eigenstate of Y.
        cirq.H(q0),  # Rotate X into Z.
        cirq.S(q1) ** -1,
        cirq.H(q1),  # Rotate Y into Z.
        cirq.measure(q0, q1, key="basis"),
    )

    sampler = CqlibSimulatorSampler(2, seed=37)
    result = sampler.run(circuit, repetitions=REPETITIONS)
    histogram = dict(result.histogram(key="basis"))
    print("Cirq X/Y-basis histogram:", histogram)
    print("compiled QCIS:\n", sampler.last_qcis[0])
    if histogram != {0: REPETITIONS}:
        raise AssertionError("X/Y-basis rotations were not preserved by compilation")
    print("PASS: Cirq basis rotations survived Cirq -> cqlib -> QCIS simulation.")


if __name__ == "__main__":
    main()
