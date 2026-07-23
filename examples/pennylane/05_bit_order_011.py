"""Verify PennyLane wire order by preparing |011> on wires [0, 1, 2]."""

from __future__ import annotations

import numpy as np
import pennylane as qml

from cqlib_adapter.pennylane import CqlibSimulatorDevice

SHOTS = 100
EXPECTED = "011"


def make_qnode(device: qml.devices.Device) -> qml.QNode:
    @qml.qnode(device, shots=SHOTS)
    def circuit():
        # PennyLane displays bits in requested wire order: wire0, wire1, wire2.
        qml.PauliX(1)
        qml.PauliX(2)
        return (
            qml.counts(wires=(0, 1, 2)),
            qml.probs(wires=(0, 1, 2)),
            qml.sample(wires=(0, 1, 2)),
        )

    return circuit


def main() -> None:
    reference = make_qnode(qml.device("default.qubit", wires=3))
    adapter_device = CqlibSimulatorDevice(wires=3)
    adapter = make_qnode(adapter_device)

    reference_counts, reference_probs, reference_samples = reference()
    adapter_counts, adapter_probs, adapter_samples = adapter()
    print("PennyLane requested wire order: (0, 1, 2)")
    print("prepared state:", EXPECTED)
    print("cqlib/canonical MSB-left storage for the same physical bits:", EXPECTED[::-1])
    print("default.qubit counts:", reference_counts)
    print("cqlib-adapter counts:", adapter_counts)
    print("cqlib-adapter probabilities:", adapter_probs)
    print("first five adapter samples:\n", adapter_samples[:5])
    print("compiled QCIS:\n", adapter_device.last_qcis[0])

    expected_probabilities = np.zeros(8)
    expected_probabilities[int(EXPECTED, 2)] = 1.0
    if reference_counts != {EXPECTED: SHOTS} or adapter_counts != {EXPECTED: SHOTS}:
        raise AssertionError("bit order is reversed or otherwise incorrect")
    np.testing.assert_array_equal(reference_probs, expected_probabilities)
    np.testing.assert_array_equal(adapter_probs, expected_probabilities)
    np.testing.assert_array_equal(reference_samples, [[0, 1, 1]] * SHOTS)
    np.testing.assert_array_equal(adapter_samples, [[0, 1, 1]] * SHOTS)
    print("PASS: canonical 110 was correctly exposed to PennyLane users as 011.")


if __name__ == "__main__":
    main()
