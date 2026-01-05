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

    # Backend configurations
    TIANYAN_HARDWARE_BACKENDS = {
        "tianyan24",
        "tianyan504", 
        "tianyan176-2",
        "tianyan176",
    }
    
    TIANYAN_SIMULATOR_BACKENDS = {
        "tianyan_sw",
        "tianyan_s", 
        "tianyan_tn",
        "tianyan_tnn",
        "tianyan_sa",
        "tianyan_swn",
    }

    # Supported operations
    SUPPORTED_OPERATIONS = {
        "Hadamard",
        "PauliX", 
        "PauliY",
        "PauliZ",
        "CNOT",
        "CZ",
        "RX",
        "RY", 
        "RZ",
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
        """Return the device name.
        
        Returns:
            String representing the device name.
        """
        return "Cqlib Quantum Device"

    @property
    def operations(self) -> Set[str]:
        """Return the set of supported operations.
        
        Returns:
            Set of supported operation names.
        """
        return self.SUPPORTED_OPERATIONS

    @property
    def backend_info(self) -> Dict[str, Any]:
        """Return information about the current backend.
        
        Returns:
            Dictionary containing backend configuration information.
        """
        return {
            "backend_type": self.machine_name,
            "is_hardware": self.machine_name in self.TIANYAN_HARDWARE_BACKENDS,
            "is_simulator": self.machine_name in self.TIANYAN_SIMULATOR_BACKENDS,
            "qubits": self.num_wires,
            "shots": self.num_shots,
        }

    @classmethod
    def capabilities(cls) -> Dict[str, Any]:
        """Return the device capabilities configuration.
        
        Returns:
            Dictionary containing supported features and capabilities.
        """
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

    def supports_operation(self, operation: Any) -> bool:
        """Check if a specific quantum operation is supported.
        
        Args:
            operation: Quantum operation to check.

        Returns:
            True if the operation is supported, False otherwise.
        """
        supported_operations = {
            "PauliX",
            "PauliY",
            "PauliZ",
            "Hadamard",
            "S",
            "T",
            "RX",
            "RY",
            "RZ",
            "CNOT",
            "CZ",
        }
        return getattr(operation, "name", None) in supported_operations

    def execute(
        self, 
        circuits: Union[QuantumScript, List[QuantumScript]], 
        execution_config: Any = None,
    ) -> List[Any]:
        """Execute quantum circuits on the device.
        
        Args:
            circuits: Single quantum circuit or list of circuits to execute.
            execution_config: Execution configuration parameters.

        Returns:
            List of execution results for each circuit.
        """
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]
        
        return [self.circuit_executor.execute_circuit(circuit) for circuit in circuits]
        
    
    def __repr__(self) -> str:
        """Return string representation of the device.
        
        Returns:
            String representation of the device.
        """
        return f"<{self.name} device (wires={self.wires}, shots={self.shots})>"
    
    def preprocess_transforms(self, execution_config: Any = None) -> Any:
        """Define the preprocessing transformation pipeline.
        
        Args:
            execution_config: Execution configuration parameters.

        Returns:
            TransformProgram: Preprocessing transformation program.
        """
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
        program.add_transform(
            qml.devices.preprocess.decompose,
            stopping_condition=self.supports_operation,
            name=self.short_name,
        )
        return program