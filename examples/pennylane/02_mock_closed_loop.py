"""No-network PennyLane Device/result contract test with preset counts.

This mock does not simulate the submitted circuit. Use the local cqlib examples
for quantum-semantic validation.
"""

import numpy as np
import pennylane as qml

from cqlib_adapter.pennylane.testing import ResultSpec, make_pennylane_device

SHOTS = 20


def main() -> None:
    # cqlib result strings are physical MSB-left: 110 means q2=1,q1=1,q0=0.
    device, cloud = make_pennylane_device(
        [ResultSpec({"110": SHOTS}, (0, 1, 2), status_ready=True)],
        wires=3,
    )

    @qml.qnode(device, shots=SHOTS)
    def circuit():
        qml.PauliX(1)
        qml.PauliX(2)
        return (
            qml.counts(wires=(0, 1, 2)),
            qml.probs(wires=(0, 1, 2)),
            qml.sample(wires=(0, 1, 2)),
        )

    counts, probabilities, samples = circuit()
    print("PennyLane counts:", counts)
    print("PennyLane probabilities:", probabilities)
    print("first five samples:\n", samples[:5])
    print("task IDs:", device.last_task_ids)
    print("submitted QCIS:\n", device.last_qcis[0])
    print("mock transport call:", cloud.calls[0])

    if counts != {"011": SHOTS}:
        raise AssertionError(f"expected PennyLane wire-order result 011, got {counts}")
    np.testing.assert_array_equal(probabilities, [0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(samples, [[0, 1, 1]] * SHOTS)
    print("PASS: mock Device/result contract succeeded; semantics not evaluated.")


if __name__ == "__main__":
    main()
