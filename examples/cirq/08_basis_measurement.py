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
