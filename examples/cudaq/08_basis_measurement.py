"""Verify explicit CUDA-Q X-basis measurement with local cqlib QCIS."""

import cudaq

from cqlib_adapter.cudaq import CqlibSimulator

SHOTS = 64


@cudaq.kernel
def x_basis_plus() -> None:
    qubit = cudaq.qubit()
    h(qubit)  # noqa: F821  # Prepare |+>.
    h(qubit)  # noqa: F821  # Rotate X into Z before mz.
    mz(qubit)  # noqa: F821


def main() -> None:
    cudaq.set_target("qpp-cpu")
    simulator = CqlibSimulator(1, seed=41)
    counts = dict(simulator.sample(x_basis_plus, shots_count=SHOTS))
    print("CUDA-Q X-basis counts:", counts)
    print("compiled QCIS:\n", simulator.last_qcis)
    if counts != {"0": SHOTS}:
        raise AssertionError("X-basis rotation was not preserved by compilation")
    print("PASS: CUDA-Q basis rotation survived OpenQASM 2 -> cqlib -> QCIS simulation.")


if __name__ == "__main__":
    main()
