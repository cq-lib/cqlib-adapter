"""Verify PennyLane observable-valued X/Y measurements with local cqlib."""

import pennylane as qml

from cqlib_adapter.pennylane import CqlibSimulatorDevice

SHOTS = 64


def main() -> None:
    device = CqlibSimulatorDevice(wires=2, seed=31)

    @qml.qnode(device, shots=SHOTS)
    def circuit() -> tuple[dict[float, int], dict[float, int]]:
        qml.Hadamard(0)  # |+>, the +1 eigenstate of X.
        qml.Hadamard(1)
        qml.S(1)  # |+i>, the +1 eigenstate of Y.
        return (
            qml.counts(qml.X(0), all_outcomes=True),
            qml.counts(qml.Y(1), all_outcomes=True),
        )

    x_counts, y_counts = circuit()
    print("PennyLane X-basis counts:", x_counts)
    print("PennyLane Y-basis counts:", y_counts)
    print("compiled QCIS:\n", device.last_qcis[0])
    expected = {1.0: SHOTS, -1.0: 0}
    if x_counts != expected or y_counts != expected:
        raise AssertionError("observable basis or eigenvalue conversion is incorrect")
    print("PASS: PennyLane Pauli X/Y observables returned the expected eigenvalues.")


if __name__ == "__main__":
    main()
