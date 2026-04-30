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

"""Quantum Circuit Executor for PennyLane Adapter.

This module provides a sophisticated circuit executor that bridges PennyLane quantum
circuits with various backend computation platforms, including local simulators,
Tianyan simulators, and Tianyan hardware devices. It handles circuit conversion,
backend initialization, execution, and measurement formatting.
"""

import json
import logging
from enum import Enum
from functools import singledispatchmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript

from cqlib import TianYanPlatform, Circuit
from cqlib.mapping import transpile_qcis
from cqlib.simulator import StatevectorSimulator
from cqlib.utils import qasm2


class BackendType(Enum):
    """Enumeration of supported quantum computation backend types."""
    LOCAL_SIMULATOR = "local"
    TIANYAN_SIMULATOR = "tianyan_simulator"
    TIANYAN_HARDWARE = "tianyan_hardware"


class CircuitExecutor:
    """Executes quantum circuits across different computational backends.

    This class abstracts the connection and execution logic for local simulation,
    cloud simulation, and actual quantum hardware provided by the Tianyan platform.
    """

    def __init__(self, device_config: Dict[str, Any]) -> None:
        """Initializes the circuit executor with the provided device configuration."""
        self.device_config = device_config
        self.logger = self._setup_logger()
        self.cqlib_backend: Optional[TianYanPlatform] = None
        self._execution_count = 0

        # Local import to prevent circular dependencies during module initialization
        from .device import CQLibDevice

        machine_name = self.device_config.get('machine_name', 'default')

        if machine_name != "default":
            login_key = self.device_config.get('login_key')
            if not login_key:
                raise ValueError(f"Login key required for backend: {machine_name}")

            # Fetch backends from API and cache in CQLibDevice class
            try:
                CQLibDevice.get_available_backends(token=login_key)
            except Exception as e:
                self.logger.error("Failed to fetch backend list: %s", e)
                raise ConnectionError(f"Could not connect to Tianyan API: {e}") from e

        self._backend_type = self._determine_backend_type()
        self._initialize_backend()

        self.logger.info("CircuitExecutor initialized with %s backend", self._backend_type.value)

    def _setup_logger(self) -> logging.Logger:
        """Configures and returns a logger instance for execution tracking."""
        logger = logging.getLogger(f"CircuitExecutor.{id(self)}")

        if self.device_config.get('verbose', False):
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            if not logger.handlers:
                logger.addHandler(handler)

        return logger

    def _determine_backend_type(self) -> BackendType:
        """Determines the appropriate backend type based on the device configuration."""
        from .device import CQLibDevice

        backend_name = self.device_config.get('machine_name', 'default')

        if backend_name == "default":
            return BackendType.LOCAL_SIMULATOR
        if backend_name in CQLibDevice.TIANYAN_HARDWARE_BACKENDS:
            return BackendType.TIANYAN_HARDWARE
        if backend_name in CQLibDevice.TIANYAN_SIMULATOR_BACKENDS:
            return BackendType.TIANYAN_SIMULATOR

        raise ValueError(f"Unknown or unsupported backend: {backend_name}")

    def _initialize_backend(self) -> None:
        """Initializes the connection to the quantum computation backend."""
        if self._backend_type == BackendType.LOCAL_SIMULATOR:
            self.logger.debug("Using local simulator - no backend connection needed.")
            return

        login_key = self.device_config.get('login_key')
        if not login_key:
            raise ValueError("Login key required for Tianyan backend authentication.")

        try:
            self.cqlib_backend = TianYanPlatform(
                login_key=login_key,
                machine_name=self.device_config['machine_name']
            )
            self.logger.info(
                "Successfully connected to Tianyan backend: %s",
                self.device_config['machine_name']
            )
        except Exception as error:
            self.logger.error(
                "Backend connection failed for %s: %s",
                self.device_config['machine_name'], error
            )
            raise ConnectionError(f"Backend connection failed: {error}") from error

    def execute_circuit(self, circuit: QuantumScript) -> Union[List[Any], Any]:
        """Executes a quantum circuit and returns the measurement results."""
        self._validate_circuit(circuit)
        self._execution_count += 1
        results = []

        for measurement in circuit.measurements:
            self._validate_measurement_support(measurement)

            cqlib_circuit, cqlib_qcis = self._convert_to_cqlib_format(circuit)
            raw_result = self._execute_on_backend(cqlib_circuit, cqlib_qcis)
            result = self._execute_measurement(measurement, raw_result)

            results.append(result)

        return results[0] if len(results) == 1 else results

    def _validate_circuit(self, circuit: QuantumScript) -> None:
        """Validates circuit constraints against the current device configuration."""
        has_state_measurement = any(
            isinstance(m, qml.measurements.StateMP) for m in circuit.measurements
        )
        shots = self.device_config.get('shots')

        if has_state_measurement and shots is not None:
            raise ValueError(
                f"State measurement requires shots=None for exact statevector simulation. "
                f"Current shots: {shots}"
            )

    def _validate_measurement_support(self, measurement: Any) -> None:
        """Validates that the active backend supports the requested measurement."""
        supported_measurements = {
            BackendType.LOCAL_SIMULATOR: {
                qml.measurements.ProbabilityMP,
                qml.measurements.ExpectationMP,
                qml.measurements.StateMP,
                qml.measurements.SampleMP
            },
            BackendType.TIANYAN_SIMULATOR: {
                qml.measurements.ProbabilityMP,
                qml.measurements.SampleMP,
                qml.measurements.ExpectationMP,
            },
            BackendType.TIANYAN_HARDWARE: {
                qml.measurements.ProbabilityMP,
                qml.measurements.SampleMP,
                qml.measurements.ExpectationMP,
            }
        }

        measurement_type = type(measurement)
        if measurement_type not in supported_measurements[self._backend_type]:
            supported_names = [m.__name__ for m in supported_measurements[self._backend_type]]
            raise ValueError(
                f"Backend {self._backend_type.value} does not support "
                f"measurement type {measurement_type.__name__}. "
                f"Supported measurements: {supported_names}"
            )

    def _convert_to_cqlib_format(self, circuit: QuantumScript) -> Tuple[Any, str]:
        """Converts a PennyLane circuit into CQLib-compatible parsed objects and QCIS string."""
        try:
            cqlib_circuit = self._build_cqlib_circuit(circuit)
            return cqlib_circuit, cqlib_circuit.qcis
        except Exception as error:
            self.logger.error("Circuit conversion from PennyLane to CQLib format failed: %s", error)
            raise ValueError(f"Circuit conversion failed: {error}") from error

    def _build_cqlib_circuit(self, circuit: QuantumScript) -> Circuit:
        """Directly constructs a CQLib Circuit object from a PennyLane quantum circuit."""
        device_wires = self.device_config.get('wires')
        if isinstance(device_wires, int):
            num_wires = device_wires
        elif device_wires is not None and hasattr(device_wires, '__len__'):
            num_wires = len(device_wires)
        else:
            num_wires = max(circuit.wires.labels) + 1 if circuit.wires else 1

        cqlib_cir = Circuit(num_wires)

        def map_operation(op, target_circuit: Circuit):
            op_name = op.name
            wires = op.wires.tolist()
            params = op.parameters

            if op_name == "X2PGate":
                target_circuit.x2p(wires[0])
            elif op_name == "X2MGate":
                target_circuit.x2m(wires[0])
            elif op_name == "Y2PGate":
                target_circuit.y2p(wires[0])
            elif op_name == "Y2MGate":
                target_circuit.y2m(wires[0])
            elif op_name == "XY2PGate":
                target_circuit.xy2p(wires[0], params[0])
            elif op_name == "XY2MGate":
                target_circuit.xy2m(wires[0], params[0])
            elif op_name == "PauliX":
                target_circuit.x(wires[0])
            elif op_name == "PauliY":
                target_circuit.y(wires[0])
            elif op_name == "PauliZ":
                target_circuit.z(wires[0])
            elif op_name == "Hadamard":
                target_circuit.h(wires[0])
            elif op_name == "RX":
                target_circuit.rx(wires[0], params[0])
            elif op_name == "RY":
                target_circuit.ry(wires[0], params[0])
            elif op_name == "RZ":
                target_circuit.rz(wires[0], params[0])
            elif op_name == "CNOT":
                target_circuit.cx(wires[0], wires[1])
            elif op_name == "CZ":
                target_circuit.cz(wires[0], wires[1])
            elif op_name == "S":
                target_circuit.s(wires[0])
            elif op_name == "T":
                target_circuit.t(wires[0])
            else:
                try:
                    decomposed_ops = op.decomposition()
                    for d_op in decomposed_ops:
                        map_operation(d_op, target_circuit)
                except Exception as e:
                    self.logger.warning(
                        f"Operation {op_name} not natively mapped and decomposition failed: {e}"
                    )

        for op in circuit.operations:
            map_operation(op, cqlib_cir)

        return cqlib_cir

    def _execute_measurement(self, measurement: Any, raw_result: Dict[str, Any]) -> Any:
        """Dispatches the raw result to the appropriate measurement processing implementation."""
        return self._execute_measurement_impl(measurement, raw_result)

    @singledispatchmethod
    def _execute_measurement_impl(self, measurement: Any, raw_result: Dict[str, Any]) -> Any:
        """Fallback implementation for unsupported measurement types."""
        raise NotImplementedError(
            f"Measurement type {type(measurement).__name__} is not supported. "
            f"Supported types: ProbabilityMP, ExpectationMP, StateMP, SampleMP"
        )

    @_execute_measurement_impl.register
    def _(
    self, measurement: qml.measurements.ProbabilityMP, raw_result: Dict[str, Any]) -> np.ndarray:
        """Extracts and formats probability distributions, supporting partial measurement."""
        probabilities = self._extract_probabilities(raw_result)

        if measurement.wires:
            device_wires = self.device_config.get('wires')
            if isinstance(device_wires, int):
                device_wire_list = list(range(device_wires))
            else:
                device_wire_list = list(device_wires)

            target_indices = [device_wire_list.index(w) for w in measurement.wires.labels]

            marginal_probs = {}
            for bitstring, prob in probabilities.items():
                sub_bitstring = "".join([bitstring[i] for i in target_indices if i < len(bitstring)])
                marginal_probs[sub_bitstring] = marginal_probs.get(sub_bitstring, 0.0) + prob

            return self._format_probabilities(marginal_probs)

        return self._format_probabilities(probabilities)

    @_execute_measurement_impl.register
    def _(self, measurement: qml.measurements.ExpectationMP, raw_result: Dict[str, Any]) -> float:
        """Calculates the expectation value for Pauli-Z observables from raw probabilities."""
        # Process probabilities via _extract_probabilities to handle endianness reversal.
        probabilities = self._extract_probabilities(raw_result)
        
        if not probabilities or not isinstance(probabilities, dict):
            raise ValueError("Execution results must contain a valid 'probabilities' dictionary")

        pauli_indices = list(measurement.obs.wires.labels)
        expectation = 0.0

        for bitstring, prob in probabilities.items():
            eigenvalue = 1
            for qubit_idx in pauli_indices:
                if qubit_idx < len(bitstring):
                    if bitstring[qubit_idx] == '1':
                        eigenvalue *= -1
            expectation += prob * eigenvalue
            
        return expectation

    @_execute_measurement_impl.register
    def _(self, measurement: qml.measurements.StateMP, raw_result: Dict[str, Any]) -> np.ndarray:
        """Extracts the statevector from simulation results."""
        if self._backend_type != BackendType.LOCAL_SIMULATOR:
            raise ValueError(
                "Statevector measurement is only supported on local simulators. "
                f"Current backend: {self._backend_type.value}"
            )
        statevector = raw_result.get('statevector')
        if statevector is None:
            raise ValueError("Statevector not found in execution results")
        
        # Ensure the statevector is returned as a dense 1D Numpy array for PennyLane compatibility.
        if isinstance(statevector, dict):
            num_qubits = len(next(iter(statevector.keys())))
            dense_state = np.zeros(2 ** num_qubits, dtype=complex)
            for bitstring, amplitude in statevector.items():
                # Reverse the dictionary keys to align with the expected endianness format.
                index = int(bitstring[::-1], 2)
                dense_state[index] = amplitude
            return dense_state

        return np.array(statevector)

    @_execute_measurement_impl.register
    def _(self, measurement: qml.measurements.SampleMP, raw_result: Dict[str, Any]) -> np.ndarray:
        """Extracts measurement samples from execution results, supporting partial measurement."""
        # Extract samples, ensuring endianness reversal and format conversion are applied.
        samples = self._extract_samples(raw_result)
        if samples is None:
            raise ValueError("No measurement samples found in execution results")
            
        if measurement.wires:
            device_wires = self.device_config.get('wires')
            if isinstance(device_wires, int):
                device_wire_list = list(range(device_wires))
            else:
                device_wire_list = list(device_wires)
                
            target_indices = [device_wire_list.index(w) for w in measurement.wires.labels]            
            return samples[:, target_indices]
            
        return samples

    def _execute_on_backend(self, cqlib_circuit: Any, cqlib_qcis: str) -> Dict[str, Any]:
        """Routes execution to the appropriate backend handler."""
        backend_handlers = {
            BackendType.LOCAL_SIMULATOR: self._execute_local_simulator,
            BackendType.TIANYAN_SIMULATOR: self._execute_tianyan_simulator,
            BackendType.TIANYAN_HARDWARE: self._execute_tianyan_hardware
        }
        handler = backend_handlers[self._backend_type]
        return handler(cqlib_circuit, cqlib_qcis)

    def _execute_local_simulator(self, cqlib_circuit: Any, cqlib_qcis: str) -> Dict[str, Any]:
        """Executes the circuit on the local statevector simulator."""
        simulator = StatevectorSimulator(cqlib_circuit)
        return {
            'probabilities': simulator.probs(),
            'samples': simulator.sample(is_raw_data=True),
            'statevector': simulator.statevector() 
        }

    def _execute_tianyan_simulator(self, cqlib_circuit: Any, cqlib_qcis: str) -> Dict[str, Any]:
        """Executes the circuit on the Tianyan cloud simulator via API."""
        cqlib_circuit.measure_all()
        query_id = self.cqlib_backend.submit_experiment(
            cqlib_circuit.qcis,
            num_shots=self.device_config.get('shots')
        )
        raw_result = self.cqlib_backend.query_experiment(query_id)[0]
        sample_res = np.array(raw_result['resultStatus'][1:])
        return {
            'probabilities': json.loads(raw_result['probability']),
            'samples': sample_res
        }

    def _execute_tianyan_hardware(self, cqlib_circuit: Any, cqlib_qcis: str) -> Dict[str, Any]:
        """Executes the circuit on physical Tianyan quantum hardware."""
        from .ext_mapping import HardwareMapper
        cqlib_circuit.measure_all()
        mapper = self.device_config.get('mapping', None)
        
        if mapper:
            hw_map = HardwareMapper(mapper)
            compiled_circuit = hw_map.map_qcis_code(cqlib_circuit.qcis)
        else:
            compiled_circuit = transpile_qcis(cqlib_qcis, self.cqlib_backend)[0]
            compiled_circuit.measure_all()
            print("transpile successfully!")
            if hasattr(compiled_circuit, 'qcis'):
                compiled_circuit = compiled_circuit.qcis

        query_id = self.cqlib_backend.submit_experiment(
            compiled_circuit,
            num_shots=self.device_config.get('shots')
        )
        raw_result = self.cqlib_backend.query_experiment(
            query_id,
            readout_calibration=True
        )
        sample_res = np.array(raw_result[0]['resultStatus'][1:])
        return {
            'probabilities': raw_result[0]['probability'],
            'samples': sample_res
        }

    def _extract_probabilities(self, raw_result: Dict[str, Any]) -> Dict[str, float]:
        """Extracts and endian-reverses the probability distribution dictionary."""
        probabilities = raw_result.get('probabilities', {})
        if probabilities and isinstance(probabilities, dict):
            return {key[::-1]: value for key, value in probabilities.items()}
        return probabilities

    def _extract_samples(self, raw_result: Dict[str, Any]) -> Any:
        """Extracts and converts samples to the format expected by PennyLane."""
        samples = raw_result.get('samples')
        # Uniformly convert sample formats for all backends to handle bit reversal via little_endian.
        if samples is not None:
            return samples_to_pennylane_format(samples, self.device_config['wires'])
        return samples

    def _format_probabilities(self, probabilities: Dict[str, float]) -> np.ndarray:
        """Converts a probability dictionary into a dense NumPy array distribution."""
        if not probabilities:
            raise ValueError("No probability distribution found in execution results")
        
        num_qubits = len(next(iter(probabilities.keys())))
        prob_array = np.zeros(2 ** num_qubits)
        
        for bitstring, prob in probabilities.items():
            index = int(bitstring, 2)
            prob_array[index] = prob
            
        return prob_array


def decimal_to_binary_array(
    decimal_value: int, num_bits: int, little_endian: bool = True
) -> np.ndarray:
    """Converts a decimal integer into a binary numpy array."""
    binary_string = np.binary_repr(int(decimal_value), width=num_bits)
    bits = np.array([int(bit) for bit in binary_string])
    return bits[::-1] if little_endian else bits


def samples_to_pennylane_format(
    samples: Union[List[int], np.ndarray],
    num_qubits: Optional[int] = None,
    measured_qubits: Optional[List[int]] = None,
    little_endian: bool = True
) -> np.ndarray:
    """Converts raw sample data into a PennyLane compatible binary matrix."""
    samples_array = np.asarray(samples)
    
    if measured_qubits is not None:
        num_bits = len(measured_qubits)
    elif num_qubits is not None:
        num_bits = num_qubits
    elif len(samples_array) == 0:
        raise ValueError("Cannot determine number of bits from empty samples array")
    else:
        max_value = np.max(samples_array)
        num_bits = int(np.ceil(np.log2(max_value + 1))) if max_value > 0 else 1

    n_shots = len(samples_array)
    binary_matrix = np.zeros((n_shots, num_bits), dtype=int)
    
    for i, sample in enumerate(samples_array):
        binary_matrix[i] = decimal_to_binary_array(sample, num_bits, little_endian)
        
    return binary_matrix