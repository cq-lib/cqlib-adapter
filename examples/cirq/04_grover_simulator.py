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

"""Compare Cirq Simulator with the real cqlib adapter on two-qubit Grover."""

from __future__ import annotations

from time import perf_counter

import cirq

from cqlib_adapter.cirq import CqlibSimulatorSampler

SHOTS = 1024


def grover_circuit() -> cirq.Circuit:
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(
        cirq.H(q0),
        cirq.H(q1),
        # Oracle marks |11>.
        cirq.CZ(q0, q1),
        # Diffusion operator.
        cirq.H(q0),
        cirq.H(q1),
        cirq.X(q0),
        cirq.X(q1),
        cirq.CZ(q0, q1),
        cirq.X(q0),
        cirq.X(q1),
        cirq.H(q0),
        cirq.H(q1),
        cirq.measure(q0, q1, key="result"),
    )


def main() -> None:
    circuit = grover_circuit()
    print("Grover circuit (marked state |11>):")
    print(circuit)

    reference = cirq.Simulator(seed=7)
    started = perf_counter()
    reference_result = reference.run(circuit, repetitions=SHOTS)
    reference_elapsed = perf_counter() - started

    adapter = CqlibSimulatorSampler(2, seed=7)
    started = perf_counter()
    adapter_result = adapter.run(circuit, repetitions=SHOTS)
    adapter_elapsed = perf_counter() - started

    reference_histogram = dict(reference_result.histogram(key="result"))
    adapter_histogram = dict(adapter_result.histogram(key="result"))
    print("Cirq Simulator histogram:", reference_histogram)
    print(f"Cirq Simulator execution time: {reference_elapsed:.6f} s")
    print("cqlib-adapter histogram:", adapter_histogram)
    print(f"adapter compile + cqlib simulation + result time: {adapter_elapsed:.6f} s")
    if reference_elapsed > 0:
        print(
            f"single-run adapter/reference time ratio: {adapter_elapsed / reference_elapsed:.3f}x"
        )
    print("Tianyan-compatible QCIS simulated by cqlib:\n", adapter.last_qcis[0])

    if reference_histogram != {3: SHOTS} or adapter_histogram != {3: SHOTS}:
        raise AssertionError("Grover did not find |11> in every shot")
    print("PASS: both Cirq paths found |11> in every shot.")
    print("Note: one short run is a functional timing comparison, not a benchmark.")


if __name__ == "__main__":
    main()
