"""Compare PennyLane default.qubit and the cqlib-adapter Grover path."""

from __future__ import annotations

from time import perf_counter

import pennylane as qml

from cqlib_adapter.pennylane import CqlibSimulatorDevice

SHOTS = 1024
EXPECTED = "11"


def grover_operations() -> None:
    """One two-qubit Grover iteration marking |11>."""

    for wire in (0, 1):
        qml.Hadamard(wire)
    qml.CZ((0, 1))
    for wire in (0, 1):
        qml.Hadamard(wire)
        qml.PauliX(wire)
    qml.CZ((0, 1))
    for wire in (0, 1):
        qml.PauliX(wire)
        qml.Hadamard(wire)


def make_qnode(device: qml.devices.Device) -> qml.QNode:
    @qml.qnode(device, shots=SHOTS)
    def circuit():
        grover_operations()
        return qml.counts(wires=(0, 1))

    return circuit


def main() -> None:
    reference = make_qnode(qml.device("default.qubit", wires=2))
    adapter_device = CqlibSimulatorDevice(wires=2)
    adapter = make_qnode(adapter_device)

    started = perf_counter()
    reference_counts = reference()
    reference_time = perf_counter() - started
    started = perf_counter()
    adapter_counts = adapter()
    adapter_time = perf_counter() - started

    print("Grover circuit (marked state |11>):")
    print(qml.draw(adapter)())
    print("PennyLane default.qubit counts:", reference_counts)
    print(f"default.qubit execution time: {reference_time:.6f} s")
    print("cqlib-adapter counts:", adapter_counts)
    print(f"adapter compile + cqlib simulation + result time: {adapter_time:.6f} s")
    print(f"single-run adapter/reference time ratio: {adapter_time / reference_time:.3f}x")
    print("Tianyan-compatible QCIS simulated by cqlib:\n", adapter_device.last_qcis[0])

    if reference_counts != {EXPECTED: SHOTS} or adapter_counts != {EXPECTED: SHOTS}:
        raise AssertionError(
            f"Grover expected {EXPECTED!r}; reference={reference_counts}, adapter={adapter_counts}"
        )
    print("PASS: both PennyLane paths found |11> in every shot.")
    print("Note: one short run is a functional timing comparison, not a benchmark.")


if __name__ == "__main__":
    main()
