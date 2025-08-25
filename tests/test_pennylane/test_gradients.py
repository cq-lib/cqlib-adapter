# test_gradients.py
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
import os
import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.devices import Device

# Configuration parameters
TOKEN = os.getenv("CQLIB_TOKEN", None)
DEFAULT_BACKEND = "default"
SHOTS = 500
WIRES = 2
STEP_SIZE = 0.1
TRAINING_STEPS = 3
INITIAL_PARAMS = np.array([0.5, 0.8])


def create_device(
    backend_name: str = DEFAULT_BACKEND, shots: int = SHOTS, wires: int = WIRES
) -> Device:
    """Create a quantum device instance.

    Args:
        backend_name (str): Name of the quantum backend to use.
        shots (int): Number of measurement shots.
        wires (int): Number of quantum wires (qubits).

    Returns:
        Device: Configured quantum device instance.
    """
    return qml.device(
        "cqlib.device",
        wires=wires,
        shots=shots,
        cqlib_backend_name=backend_name,
        login_key=TOKEN if backend_name != "default" else None,
    )


def create_circuit(device: Device, diff_method: str = "parameter-shift") -> qml.QNode:
    """Create a differentiable quantum circuit.

    Args:
        device (Device): Quantum device to run the circuit on.
        diff_method (str): Differentiation method.

    Returns:
        qml.QNode: Configured quantum circuit.
    """

    @qml.qnode(device, diff_method=diff_method)
    def circuit(params: np.ndarray) -> float:
        """Parameterized quantum circuit.

        Args:
            params (np.ndarray): Rotation parameters [theta_x, theta_y].

        Returns:
            float: Expectation value of PauliZ on wire 0.
        """
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    return circuit


@pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff"])
def test_gradient_computation(diff_method):
    """Test that gradients can be computed using different methods.
    
    Args:
        diff_method (str): Differentiation method to test.
    """
    device = create_device(DEFAULT_BACKEND)
    circuit = create_circuit(device, diff_method=diff_method)
    
    # Test gradient computation
    grad_fn = qml.grad(circuit)
    gradient = grad_fn(INITIAL_PARAMS)
    
    # Assertions
    assert gradient is not None
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == INITIAL_PARAMS.shape


def test_optimization_converges():
    """Test that gradient descent optimization reduces expectation value."""
    device = create_device(DEFAULT_BACKEND)
    circuit = create_circuit(device)

    optimizer = qml.GradientDescentOptimizer(stepsize=STEP_SIZE)
    params = INITIAL_PARAMS.copy()
    initial_expval = circuit(params)

    for _ in range(TRAINING_STEPS):
        params = optimizer.step(circuit, params)

    final_expval = circuit(params)

    # Assertion: final expectation value should not exceed the initial value
    assert final_expval <= initial_expval + 1e-6


def test_circuit_differentiability():
    """Test that the circuit is properly differentiable."""
    device = create_device(DEFAULT_BACKEND)
    circuit = create_circuit(device)
    
    # Test that we can compute the gradient
    try:
        grad_fn = qml.grad(circuit)
        gradient = grad_fn(INITIAL_PARAMS)
        assert gradient is not None
    except Exception as e:
        pytest.fail(f"Circuit differentiation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])