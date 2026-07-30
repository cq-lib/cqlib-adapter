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

"""Verify native CUDA-Q X/Y-basis measurements with local cqlib QCIS."""

import cudaq

from cqlib_adapter.cudaq import CqlibSimulator

SHOTS = 64


@cudaq.kernel
def x_and_y_basis() -> None:
    qubits = cudaq.qvector(2)
    h(qubits[0])  # noqa: F821  # Prepare |+>.
    h(qubits[1])  # noqa: F821
    s(qubits[1])  # noqa: F821  # Prepare |+i>.
    mx(qubits[0])  # noqa: F821
    my(qubits[1])  # noqa: F821


def main() -> None:
    cudaq.set_target("qpp-cpu")
    simulator = CqlibSimulator(2, seed=41)
    counts = dict(simulator.sample(x_and_y_basis, shots_count=SHOTS))
    print("CUDA-Q X/Y-basis counts:", counts)
    print("compiled QCIS:\n", simulator.last_qcis)
    if counts != {"00": SHOTS}:
        raise AssertionError("CUDA-Q mx/my basis measurements were not preserved")
    print("PASS: CUDA-Q mx/my survived Quake MLIR -> cqlib -> QCIS simulation.")


if __name__ == "__main__":
    main()
