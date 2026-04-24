# This code is part of cqlib.
#
# Copyright (C) 2026 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""PennyLane device implementation using CQLib backend.

This module provides a custom quantum device that interfaces between PennyLane
and various CQLib backends, including local simulators and TianYan cloud services.
"""

import os
from typing import Any, Dict, List, Set, Union

import pennylane as qml
from pennylane.devices import Device
from pennylane.tape import QuantumScript, QuantumScriptOrBatch

from .circuit_executor import CircuitExecutor
from .native_gates import (
    X2PGate, X2MGate,
    Y2PGate, Y2MGate,
    XY2PGate, XY2MGate
)
from ..utils.api_client import ApiClient


class CQLibDevice(Device):
    """Custom quantum device implementing PennyLane device interface using CQLib backend.

    This device provides an interface between PennyLane and various CQLib backends,
    including local simulators, TianYan cloud simulators, and TianYan hardware.

    Attributes:
        short_name (str): Short identifier for the device.
        config_filepath (str): Path to device configuration file.
        TIANYAN_HARDWARE_BACKENDS (set): Set of supported hardware backend names.
        TIANYAN_SIMULATOR_BACKENDS (set): Set of supported simulator backend names.
        SUPPORTED_OPERATIONS (set): Set of supported quantum operations.
    """

    # Backend configurations - dynamically populated from API
    # Empty sets, populated on first fetch via get_available_backends()
    TIANYAN_HARDWARE_BACKENDS: Set[str] = set()
    TIANYAN_SIMULATOR_BACKENDS: Set[str] = set()

    # Supported operations
    # Updated to include custom native gates (X2P, Y2P, Rxy, etc.)
    SUPPORTED_OPERATIONS = {
        # Standard Gates
        "Hadamard",
        "PauliX",
        "PauliY",
        "PauliZ",
        "CNOT",
        "CZ",
        "RX",
        "RY",
        "RZ",
        "S",
        "T",

        "X2PGate",
        "X2MGate",
        "Y2PGate",
        "Y2MGate",
        "XY2PGate",
        "XY2MGate",
    }

    # Device metadata
    short_name = "cqlib.device"
    config_filepath = os.path.join(os.path.dirname(__file__), "cqlib_config.toml")

    def __init__(
        self,
        wires: int,
        shots: int = None,
        cqlib_backend_name: str = "default",
        login_key: str = None,
        mapping: Any = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the CQLib device.

        Args:
            wires: Number of qubits in the device.
            shots: Number of measurement shots. If None, uses analytic mode.
            cqlib_backend_name: Name of the CQLib backend to use.
            login_key: Authentication key for cloud services.
            mapping: Qubit mapping configuration.
            verbose: Whether to enable verbose output.

        Raises:
            ValueError: If invalid configuration parameters are provided.
        """
        super().__init__(wires=wires, shots=shots)

        device_config = {
            "wires": wires,
            "shots": shots,
            "machine_name": cqlib_backend_name,
            "login_key": login_key,
            "mapping": mapping,
            "verbose": verbose,
        }
        self.machine_name = cqlib_backend_name
        self.num_wires = wires
        self.num_shots = shots
        self.circuit_executor = CircuitExecutor(device_config)

    @property
    def name(self) -> str:
        """Return the device name."""
        return "Cqlib Quantum Device"

    @property
    def operations(self) -> Set[str]:
        """Return the set of supported operations."""
        return self.SUPPORTED_OPERATIONS

    @property
    def backend_info(self) -> Dict[str, Any]:
        """Return information about the current backend."""
        return {
            "backend_type": self.machine_name,
            "is_hardware": self.machine_name in self.TIANYAN_HARDWARE_BACKENDS,
            "is_simulator": self.machine_name in self.TIANYAN_SIMULATOR_BACKENDS,
            "qubits": self.num_wires,
            "shots": self.num_shots,
        }

    @classmethod
    def capabilities(cls) -> Dict[str, Any]:
        """Return the device capabilities configuration."""
        capabilities = super().capabilities().copy()

        capabilities.update(
            model="qubit",
            supports_inverse_operations=False,
            supports_analytic_computation=False,
            supports_finite_shots=True,
            returns_state=True,
            passthru_devices={
                "autograd": "default.qubit.autograd",
                "tf": "default.qubit.tf",
                "torch": "default.qubit.torch",
                "jax": "default.qubit.jax",
            },
        )
        return capabilities

    @classmethod
    def fetch_available_backends(cls, token: str = None) -> Dict[str, List[str]]:
        """Fetch available backends from TianYan API and cache them.

        Args:
            token: API token. If None, uses CQLIB_TOKEN env var.

        Returns:
            Dict with 'hardware' and 'simulator' keys containing lists of backend names.
        """
        if token is None:
            token = os.environ.get("CQLIB_TOKEN", "")

        if not token:
            raise ValueError("API token is required to fetch available backends")

        client = ApiClient(token=token)
        backends = client.get_backends()

        hardware = set()
        simulator = set()

        for backend in backends:
            code = backend.get('code')
            label = backend.get('labels')
            if code:
                if label == '1':
                    hardware.add(code)
                else:
                    simulator.add(code)

        cls.TIANYAN_HARDWARE_BACKENDS = hardware
        cls.TIANYAN_SIMULATOR_BACKENDS = simulator

        return {
            'hardware': sorted(hardware),
            'simulator': sorted(simulator)
        }

    @classmethod
    def get_available_backends(cls, token: str = None, force_refresh: bool = False) -> Dict[str, List[str]]:
        """Get available backends, fetching from API if not cached.

        Args:
            token: API token. If None, uses CQLIB_TOKEN env var.
            force_refresh: If True, force re-fetch from API.

        Returns:
            Dict with 'hardware' and 'simulator' keys containing lists of backend names.
        """
        if force_refresh or not cls.TIANYAN_HARDWARE_BACKENDS:
            cls.fetch_available_backends(token)

        return {
            'hardware': sorted(cls.TIANYAN_HARDWARE_BACKENDS),
            'simulator': sorted(cls.TIANYAN_SIMULATOR_BACKENDS)
        }

    def supports_operation(self, operation: Any) -> bool:
        """Check if a specific quantum operation is supported.

        This method is critical for preventing PennyLane from decomposing
        our native gates (like X2PGate) into standard gates.
        """
        return getattr(operation, "name", None) in self.SUPPORTED_OPERATIONS

    def execute(
        self,
        circuits: Union[QuantumScript, List[QuantumScript]],
        execution_config: Any = None,
    ) -> List[Any]:
        """Execute quantum circuits on the device."""
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]

        return [self.circuit_executor.execute_circuit(circuit) for circuit in circuits]

    def __repr__(self) -> str:
        return f"<{self.name} device (wires={self.wires}, shots={self.shots})>"

    def preprocess_transforms(self, execution_config: Any = None) -> Any:
        """Define the preprocessing transformation pipeline."""
        program = qml.transforms.core.TransformProgram()
        program.add_transform(
            qml.devices.preprocess.validate_device_wires,
            wires=self.wires,
            name=self.short_name,
        )
        program.add_transform(
            qml.devices.preprocess.validate_measurements,
            name=self.short_name,
        )

        # IMPORTANT: stopping_condition uses self.supports_operation
        # to preserve our custom native gates.
        program.add_transform(
            qml.devices.preprocess.decompose,
            stopping_condition=self.supports_operation,
            name=self.short_name,
        )
        return program
