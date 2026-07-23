"""Verify Cirq measurement order by preparing |011> on q0, q1, q2."""

from __future__ import annotations

import cirq
import numpy as np

from cqlib_adapter.cirq import CqlibSimulatorSampler

SHOTS = 100
EXPECTED_BITS = [False, True, True]
EXPECTED_INTEGER = 3


def circuit_011() -> cirq.Circuit:
    q0, q1, q2 = cirq.LineQubit.range(3)
    return cirq.Circuit(
        cirq.X(q1),
        cirq.X(q2),
        cirq.measure(q0, q1, q2, key="state"),
    )


def main() -> None:
    circuit = circuit_011()
    reference = cirq.Simulator(seed=11).run(circuit, repetitions=SHOTS)
    adapter_sampler = CqlibSimulatorSampler(3, seed=11)
    adapter = adapter_sampler.run(circuit, repetitions=SHOTS)

    print("Cirq measurement qubit order: (q0, q1, q2)")
    print("prepared state: 011")
    print("cqlib canonical MSB-left storage for the same bits: 110")
    print("Cirq Simulator histogram:", reference.histogram(key="state"))
    print("cqlib-adapter histogram:", adapter.histogram(key="state"))
    print("first five adapter measurement rows:\n", adapter.measurements["state"][:5])
    print("compiled QCIS:\n", adapter_sampler.last_qcis[0])

    if reference.histogram(key="state") != {EXPECTED_INTEGER: SHOTS}:
        raise AssertionError("Cirq Simulator produced an unexpected bit order")
    if adapter.histogram(key="state") != {EXPECTED_INTEGER: SHOTS}:
        raise AssertionError("cqlib-adapter reversed the Cirq measurement order")
    np.testing.assert_array_equal(adapter.measurements["state"], [EXPECTED_BITS] * SHOTS)
    print("PASS: canonical 110 is exposed as Cirq measurement bits 011 and integer 3.")


if __name__ == "__main__":
    main()
