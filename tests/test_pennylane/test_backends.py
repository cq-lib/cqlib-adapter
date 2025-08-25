# test_backends.py
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
SHOTS = 500
WIRES = 2
INITIAL_PARAMS = np.array([0.5, 0.8])


def create_device(
    backend_name: str, shots: int = SHOTS, wires: int = WIRES
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


@pytest.mark.parametrize("backend_name", ["tianyan504", "tianyan_sw", "default"])
def test_backend_runs(backend_name):
    """Test that circuits can run successfully on a given backend.

    Args:
        backend_name (str): The backend to test.
    """
    device = create_device(backend_name)

    @qml.qnode(device)
    def circuit_probs(params: np.ndarray) -> np.ndarray:
        """Circuit returning probability distribution."""
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    @qml.qnode(device)
    def circuit_expval(params: np.ndarray) -> float:
        """Circuit returning expectation value."""
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    params = INITIAL_PARAMS.copy()
    probs = circuit_probs(params)
    expval = circuit_expval(params)

    # Assertions
    assert np.isclose(np.sum(probs), 1.0, atol=1e-6)
    assert isinstance(expval, (float, np.floating, np.ndarray))

