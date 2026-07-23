"""CUDA-Q kernel -> OpenQASM 2 -> cqlib Circuit -> native QCIS."""

from __future__ import annotations

import cudaq

from cqlib_adapter.common import CompilationOptions
from cqlib_adapter.cudaq import compile_cudaq_kernel, cudaq_to_cqlib, cudaq_to_openqasm

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
    qasm = cudaq_to_openqasm(bell_without_mz)
    bundle = cudaq_to_cqlib(bell_without_mz)
    artifact = compile_cudaq_kernel(
        bell_without_mz,
        options=CompilationOptions(target_basis=NATIVE_BASIS, seed=31),
    )

    print("CUDA-Q OpenQASM 2 before adapter measurement completion:\n", qasm)
    print("auto measurement added:", bundle.metadata.extras["auto_measure_all"])
    print("measurement slots:", bundle.metadata.measurements.slots)
    print("Tianyan-compatible QCIS:\n", artifact.qcis)
    if artifact.qcis.count("M Q") != 2:
        raise AssertionError("adapter did not add final measurement of both qubits")
    print("PASS: CUDA-Q -> OpenQASM 2 -> cqlib -> QCIS conversion succeeded.")


if __name__ == "__main__":
    main()
