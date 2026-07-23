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

"""Compare Grover search on Qiskit and the cqlib-adapter simulator backend.

Both paths run locally. The adapter path is:
Qiskit -> cqlib-adapter BackendV2 -> cqlib compile -> QCIS -> cqlib Statevector
-> TianyanJob -> standard Qiskit Result. No API key or device name is needed.
"""

from __future__ import annotations

from time import perf_counter

from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator

from cqlib_adapter.qiskit import CqlibSimulatorBackend

SHOTS = 1024
SEED = 19
EXPECTED = "11"


def grover_circuit() -> QuantumCircuit:
    """Build one two-qubit Grover iteration marking the state |11>."""

    circuit = QuantumCircuit(2, 2, name="grover-mark-11")
    circuit.h([0, 1])

    # Phase oracle: only |11> receives a minus sign.
    circuit.cz(0, 1)

    # Diffuser: reflection about the uniform superposition.
    circuit.h([0, 1])
    circuit.x([0, 1])
    circuit.cz(0, 1)
    circuit.x([0, 1])
    circuit.h([0, 1])
    circuit.measure([0, 1], [0, 1])

    # circuit.x([0,1])
    # circuit.measure([0, 1], [0, 1])

    return circuit


def run_qiskit(circuit: QuantumCircuit) -> tuple[dict[str, int], float]:
    """Transpile and execute with Qiskit's built-in reference simulator."""

    backend = BasicSimulator()
    started = perf_counter()
    executable = transpile(circuit, backend=backend, seed_transpiler=SEED)
    result = backend.run(
        executable,
        shots=SHOTS,
        seed_simulator=SEED,
    ).result()
    return dict(result.get_counts()), perf_counter() - started


def run_adapter(circuit: QuantumCircuit) -> tuple[dict[str, int], float, str]:
    """Compile and execute through BackendV2 with cqlib Statevector."""

    backend = CqlibSimulatorBackend(circuit.num_qubits)
    started = perf_counter()
    job = backend.run(circuit, shots=SHOTS, seed=SEED)
    result = job.result(timeout=30, poll_interval=0.01)
    elapsed = perf_counter() - started
    return dict(result.get_counts()), elapsed, job.qcis[0]


def main() -> None:
    circuit = grover_circuit()
    qiskit_counts, qiskit_time = run_qiskit(circuit)
    adapter_counts, adapter_time, qcis = run_adapter(circuit)

    print("Grover circuit (marked state |11>):")
    print(circuit.draw(output="text"))
    print("Qiskit BasicSimulator counts:", qiskit_counts)
    print(f"Qiskit transpile + execution time: {qiskit_time:.6f} s")
    print("cqlib-adapter Backend counts:", adapter_counts)
    print(f"Adapter compile + simulation + result time: {adapter_time:.6f} s")
    print(f"Single-run adapter/Qiskit time ratio: {adapter_time / qiskit_time:.3f}x")
    print("Tianyan-compatible QCIS simulated locally:\n", qcis)

    if set(qiskit_counts) != {EXPECTED} or set(adapter_counts) != {EXPECTED}:
        raise AssertionError(
            f"Grover expected only {EXPECTED!r}; Qiskit={qiskit_counts}, adapter={adapter_counts}"
        )
    if sum(qiskit_counts.values()) != SHOTS or sum(adapter_counts.values()) != SHOTS:
        raise AssertionError("Grover counts do not sum to SHOTS")
    print("PASS: both paths found |11> in every shot.")
    print("Note: one short run is a functional timing comparison, not a benchmark.")


if __name__ == "__main__":
    main()
