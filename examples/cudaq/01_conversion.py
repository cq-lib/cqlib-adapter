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

"""CUDA-Q kernel -> Quake MLIR -> cqlib Circuit -> native QCIS."""

from __future__ import annotations

import cudaq

from cqlib_adapter.common import CompilationOptions
from cqlib_adapter.cudaq import compile_cudaq_kernel, cudaq_to_cqlib

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


@cudaq.kernel
def bell_without_mz() -> None:
    qubits = cudaq.qvector(2)
    h(qubits[0])  # noqa: F821
    x.ctrl(qubits[0], qubits[1])  # noqa: F821


def main() -> None:
    bundle = cudaq_to_cqlib(bell_without_mz)
    artifact = compile_cudaq_kernel(
        bell_without_mz,
        options=CompilationOptions(target_basis=NATIVE_BASIS, seed=31),
    )

    print("source IR:", bundle.metadata.extras["source_ir"])
    print("Quake entrypoint:", bundle.metadata.extras["quake_entrypoint"])
    print("auto measurement added:", bundle.metadata.extras["auto_measure_all"])
    print("measurement slots:", bundle.metadata.measurements.slots)
    print("Tianyan-compatible QCIS:\n", artifact.qcis)
    if bundle.metadata.extras["source_ir"] != "quake":
        raise AssertionError("adapter did not use the direct Quake conversion path")
    if artifact.qcis.count("M Q") != 2:
        raise AssertionError("adapter did not add final measurement of both qubits")
    print("PASS: CUDA-Q -> Quake MLIR -> cqlib -> QCIS conversion succeeded.")


if __name__ == "__main__":
    main()
