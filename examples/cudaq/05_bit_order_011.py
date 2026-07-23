"""Verify CUDA-Q q0-left bit order with q0=0, q1=1, q2=1."""

from __future__ import annotations

import cudaq

from cqlib_adapter.cudaq import CqlibSimulator

SHOTS = 100


def native_counts(result: object) -> dict[str, int]:
    return {bitstring: int(result[bitstring]) for bitstring in result}  # type: ignore[index,operator]


@cudaq.kernel
def state_011() -> None:
    qubits = cudaq.qvector(3)
    x(qubits[1])  # noqa: F821
    x(qubits[2])  # noqa: F821
    mz(qubits)  # noqa: F821


def main() -> None:
    cudaq.set_target("qpp-cpu")
    reference = cudaq.sample(state_011, shots_count=SHOTS)
    simulator = CqlibSimulator(3, seed=11)
    adapter = simulator.sample(state_011, shots_count=SHOTS)

    print("prepared CUDA-Q bits (q0, q1, q2): 011")
    print("canonical cqlib MSB-left storage: 110")
    reference_counts = native_counts(reference)
    print("CUDA-Q qpp-cpu counts:", reference_counts)
    print("cqlib-adapter CUDA-Q counts:", dict(adapter))
    print("compiled QCIS:\n", simulator.last_qcis)
    if reference_counts != {"011": SHOTS}:
        raise AssertionError("CUDA-Q qpp-cpu produced an unexpected bit order")
    if dict(adapter) != {"011": SHOTS}:
        raise AssertionError("adapter did not reverse canonical storage for CUDA-Q")
    print("PASS: canonical 110 is exposed to CUDA-Q users as 011.")


if __name__ == "__main__":
    main()
