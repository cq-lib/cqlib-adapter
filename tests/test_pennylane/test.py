# This code is part of cqlib.
#
# Copyright (C) 2025 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import pennylane as qml
from pennylane import numpy as np
from pennylane.devices import Device
# Configuration parameters
TOKEN = "your_token"


DEFAULT_BACKEND = "default"
SHOTS = 500
WIRES = 2
STEP_SIZE = 0.1
TRAINING_STEPS = 10
INITIAL_PARAMS = np.array([0.5, 0.8])


def create_device(
    backend_name: str, shots: int = SHOTS, wires: int = WIRES
) -> Device:
    """Creates a quantum device instance.

    Args:
        backend_name: Name of the quantum backend to use.
        shots: Number of measurement shots. Defaults to SHOTS.
        wires: Number of quantum wires (qubits). Defaults to WIRES.

    Returns:
        qml.Device: Configured quantum device instance.
    """
    return qml.device(
        "cqlib.device",
        wires=wires,
        shots=shots,
        cqlib_backend_name=backend_name,
        login_key=TOKEN if backend_name != "default" else None,
    )


def create_circuit(
    device: Device, diff_method: str = "parameter-shift"
) -> qml.QNode:
    """Creates a differentiable quantum circuit.

    Args:
        device: Quantum device to run the circuit on.
        diff_method: Differentiation method to use. Defaults to "parameter-shift".

    Returns:
        qml.QNode: Configured quantum circuit as a QNode.
    """
    @qml.qnode(device, diff_method=diff_method)
    def circuit(params: np.ndarray) -> float:
        """Quantum circuit with parameterized rotations and measurement.

        Args:
            params: Array of rotation parameters [theta_x, theta_y].

        Returns:
            float: Expectation value of PauliZ on wire 0.
        """
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    return circuit


def test_backend(backend_name: str) -> None:
    """Tests quantum computing functionality on a specific backend.

    Args:
        backend_name: Name of the backend to test.
    """
    print(f"\n=== Testing backend: {backend_name} ===")

    # Create device
    device = create_device(backend_name)

    # Define probability measurement circuit
    @qml.qnode(device)
    def circuit_probs(params: np.ndarray) -> np.ndarray:
        """Circuit for measuring probability distribution.

        Args:
            params: Array of rotation parameters [theta_x, theta_y].

        Returns:
            np.ndarray: Probability distribution over computational basis states.
        """
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    # Define expectation value measurement circuit
    @qml.qnode(device)
    def circuit_expval(params: np.ndarray) -> float:
        """Circuit for measuring expectation value.

        Args:
            params: Array of rotation parameters [theta_x, theta_y].

        Returns:
            float: Expectation value of PauliZ on wire 0.
        """
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    # Test circuits
    params = INITIAL_PARAMS.copy()
    probabilities = circuit_probs(params)
    expectation_value = circuit_expval(params)

    print(f"Probability distribution: {probabilities}")
    print(f"Expectation value: {expectation_value:.6f}")


def optimize_circuit() -> None:
    """Performs parameter optimization of the quantum circuit.

    Uses gradient descent optimization to minimize the expectation value
    of the quantum circuit with respect to the rotation parameters.
    """
    print("\n=== Starting parameter optimization ===")

    # Create default device and initialize circuit
    device = create_device(DEFAULT_BACKEND)
    circuit = create_circuit(device)

    # Initialize optimizer and parameters
    optimizer = qml.GradientDescentOptimizer(stepsize=STEP_SIZE)
    params = INITIAL_PARAMS.copy()

    # Execute optimization loop
    for step in range(TRAINING_STEPS):
        params = optimizer.step(circuit, params)
        expectation = circuit(params)

        print(
            f"Step {step + 1:2d}: "
            f"params = [{params[0]:.6f}, {params[1]:.6f}], "
            f"expectation = {expectation:.6f}"
        )

    print("\n=== Optimization completed ===")
    print(f"Final parameters: [{params[0]:.6f}, {params[1]:.6f}]")
    print(f"Final expectation value: {circuit(params):.6f}")


def main() -> None:
    """Main function to demonstrate quantum circuit functionality.

    Tests different quantum backends and performs parameter optimization.
    """
    # Test different backends
    test_backend("tianyan504")
    test_backend("tianyan_sw")
    test_backend("default")

    # Execute optimization
    optimize_circuit()


if __name__ == "__main__":
    main()