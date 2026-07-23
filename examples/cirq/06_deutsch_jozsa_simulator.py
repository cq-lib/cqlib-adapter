"""Compare Cirq Simulator and cqlib-adapter on Deutsch-Jozsa."""

from __future__ import annotations

from time import perf_counter

import cirq

from cqlib_adapter.cirq import CqlibSimulatorSampler

SHOTS = 512


def deutsch_jozsa_balanced() -> cirq.Circuit:
    q0, q1, ancilla = cirq.LineQubit.range(3)
    return cirq.Circuit(
        cirq.X(ancilla),
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(ancilla),
        # Balanced oracle f(x0, x1) = x0 XOR x1.
        cirq.CNOT(q0, ancilla),
        cirq.CNOT(q1, ancilla),
        cirq.H(q0),
        cirq.H(q1),
        cirq.measure(q0, q1, key="input"),
    )


def main() -> None:
    circuit = deutsch_jozsa_balanced()
    print("Deutsch-Jozsa circuit (balanced XOR oracle):")
    print(circuit)

    started = perf_counter()
    reference = cirq.Simulator(seed=23).run(circuit, repetitions=SHOTS)
    reference_elapsed = perf_counter() - started

    adapter_sampler = CqlibSimulatorSampler(3, seed=23)
    started = perf_counter()
    adapter = adapter_sampler.run(circuit, repetitions=SHOTS)
    adapter_elapsed = perf_counter() - started

    reference_histogram = dict(reference.histogram(key="input"))
    adapter_histogram = dict(adapter.histogram(key="input"))
    print("Cirq Simulator histogram:", reference_histogram)
    print(f"Cirq Simulator execution time: {reference_elapsed:.6f} s")
    print("cqlib-adapter histogram:", adapter_histogram)
    print(f"adapter compile + cqlib simulation + result time: {adapter_elapsed:.6f} s")
    print("Tianyan-compatible QCIS simulated by cqlib:\n", adapter_sampler.last_qcis[0])

    if reference_histogram != {3: SHOTS} or adapter_histogram != {3: SHOTS}:
        raise AssertionError("Deutsch-Jozsa balanced oracle should return input bits 11")
    print("PASS: both Cirq paths classified the balanced oracle correctly.")


if __name__ == "__main__":
    main()
