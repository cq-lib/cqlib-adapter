"""Compare Deutsch-Jozsa on Qiskit and the cqlib-adapter simulator backend.

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
SEED = 23
EXPECTED = "1"


def deutsch_jozsa_circuit() -> QuantumCircuit:
    """Build one-input DJ for the balanced oracle f(x)=x."""

    circuit = QuantumCircuit(2, 1, name="deutsch-jozsa-balanced")
    input_qubit = 0
    ancilla = 1

    circuit.x(ancilla)
    circuit.h([input_qubit, ancilla])

    # Balanced oracle U_f: |x,y> -> |x,y xor x>.
    circuit.cx(input_qubit, ancilla)

    circuit.h(input_qubit)
    circuit.measure(input_qubit, 0)
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
    circuit = deutsch_jozsa_circuit()
    qiskit_counts, qiskit_time = run_qiskit(circuit)
    adapter_counts, adapter_time, qcis = run_adapter(circuit)

    print("Deutsch-Jozsa circuit (balanced oracle f(x)=x):")
    print(circuit.draw(output="text"))
    print("Qiskit BasicSimulator counts:", qiskit_counts)
    print(f"Qiskit transpile + execution time: {qiskit_time:.6f} s")
    print("cqlib-adapter Backend counts:", adapter_counts)
    print(f"Adapter compile + simulation + result time: {adapter_time:.6f} s")
    print(f"Single-run adapter/Qiskit time ratio: {adapter_time / qiskit_time:.3f}x")
    print("Tianyan-compatible QCIS simulated locally:\n", qcis)

    if set(qiskit_counts) != {EXPECTED} or set(adapter_counts) != {EXPECTED}:
        raise AssertionError(
            f"Deutsch-Jozsa expected only {EXPECTED!r}; "
            f"Qiskit={qiskit_counts}, adapter={adapter_counts}"
        )
    if sum(qiskit_counts.values()) != SHOTS or sum(adapter_counts.values()) != SHOTS:
        raise AssertionError("Deutsch-Jozsa counts do not sum to SHOTS")
    print("PASS: both paths identified the balanced oracle f(x)=x in every shot.")
    print("Note: one short run is a functional timing comparison, not a benchmark.")


if __name__ == "__main__":
    main()
