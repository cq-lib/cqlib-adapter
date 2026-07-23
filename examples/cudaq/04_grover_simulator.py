"""Compare CUDA-Q qpp-cpu with the cqlib adapter on two-qubit Grover."""

from __future__ import annotations

from time import perf_counter

import cudaq

from cqlib_adapter.cudaq import CqlibSimulator

SHOTS = 1024


def native_counts(result: object) -> dict[str, int]:
    return {bitstring: int(result[bitstring]) for bitstring in result}  # type: ignore[index,operator]


@cudaq.kernel
def grover_11() -> None:
    qubits = cudaq.qvector(2)
    h(qubits)  # noqa: F821
    z.ctrl(qubits[0], qubits[1])  # noqa: F821
    h(qubits)  # noqa: F821
    x(qubits)  # noqa: F821
    z.ctrl(qubits[0], qubits[1])  # noqa: F821
    x(qubits)  # noqa: F821
    h(qubits)  # noqa: F821
    mz(qubits)  # noqa: F821


def main() -> None:
    cudaq.set_target("qpp-cpu")
    started = perf_counter()
    reference = cudaq.sample(grover_11, shots_count=SHOTS)
    reference_elapsed = perf_counter() - started

    simulator = CqlibSimulator(2, seed=7)
    started = perf_counter()
    adapter = simulator.sample(grover_11, shots_count=SHOTS)
    adapter_elapsed = perf_counter() - started

    reference_counts = native_counts(reference)
    print("CUDA-Q qpp-cpu counts:", reference_counts)
    print(f"qpp-cpu time: {reference_elapsed:.6f} s")
    print("cqlib-adapter counts:", dict(adapter))
    print(f"adapter time: {adapter_elapsed:.6f} s")
    print("compiled native QCIS:\n", simulator.last_qcis)
    if reference_counts != {"11": SHOTS} or dict(adapter) != {"11": SHOTS}:
        raise AssertionError("Grover did not find |11> in every shot")
    print("PASS: CUDA-Q and cqlib paths both found |11>.")


if __name__ == "__main__":
    main()
