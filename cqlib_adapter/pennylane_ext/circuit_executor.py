"""Quantum Circuit Executor for PennyLane Adapter.

This module provides a sophisticated circuit executor that bridges PennyLane quantum
circuits with various backend computation platforms, including local simulators,
Tianyan simulators, and Tianyan hardware devices.

Key Features:
    - Support for multiple measurement types: expectation values, probabilities, 
      statevectors, and samples
    - Backend-aware execution with automatic capability validation
    - Elegant error handling and comprehensive logging
    - Seamless integration with PennyLane's quantum tape system
"""

import logging
from enum import Enum
from functools import singledispatchmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pennylane as qml
import json
from cqlib import TianYanPlatform
from cqlib.mapping import transpile_qcis
from cqlib.simulator import StatevectorSimulator
from cqlib.utils import qasm2
from pennylane.tape import QuantumScript
from pennylane.io import to_openqasm



class BackendType(Enum):
    """Enumeration of supported quantum computation backend types.
    
    Attributes:
        LOCAL_SIMULATOR: Local statevector simulator with full measurement support
        TIANYAN_SIMULATOR: Tianyan cloud-based simulator supporting probabilities and samples
        TIANYAN_HARDWARE: Physical quantum hardware supporting sample measurements only
    """
    LOCAL_SIMULATOR = "local"
    TIANYAN_SIMULATOR = "tianyan_simulator" 
    TIANYAN_HARDWARE = "tianyan_hardware"


class CircuitExecutor:
    """Executes quantum circuits across different computational backends.
    
    This class serves as the core execution engine for the PennyLane adapter,
    providing a unified interface for running quantum circuits on various
    backend platforms while handling measurement-specific transformations
    and result processing.

    Args:
        device_config: Configuration dictionary containing backend settings.
            Required keys:
            - machine_name: Backend identifier ('default' for local simulator)
            - login_key: Authentication key for Tianyan platforms (if applicable)
            - shots: Number of measurement shots (None for statevector simulations)
            - wires: Number of qubits in the system
            - verbose: Enable verbose logging if True

    Raises:
        ConnectionError: If backend connection initialization fails
        ValueError: If device configuration is invalid or incomplete

    Example:
        >>> config = {
        ...     'machine_name': 'default',
        ...     'shots': 1000,
        ...     'wires': 2,
        ...     'verbose': True
        ... }
        >>> executor = CircuitExecutor(config)
        >>> result = executor.execute_circuit(quantum_tape)
    """

    #: Set of Tianyan hardware backend identifiers
    TIANYAN_HARDWARE_BACKENDS = {
        "tianyan24", "tianyan504", "tianyan176-2", "tianyan176"
    }
    
    #: Set of Tianyan simulator backend identifiers  
    TIANYAN_SIMULATOR_BACKENDS = {
        "tianyan_sw", "tianyan_s", "tianyan_tn", 
        "tianyan_tnn", "tianyan_sa", "tianyan_swn"
    }

    def __init__(self, device_config: Dict[str, Any]) -> None:
        """Initialize the circuit executor with device configuration.
        
        The initialization process includes:
        1. Setting up logging infrastructure
        2. Determining backend type from configuration
        3. Initializing backend connection (for remote platforms)
        4. Preparing measurement processing components

        Args:
            device_config: Dictionary containing all necessary configuration
                parameters for backend operation and circuit execution.

        Raises:
            ConnectionError: If remote backend connection cannot be established
            ValueError: If required configuration parameters are missing
        """
        self.device_config = device_config
        self.logger = self._setup_logger()
        self.cqlib_backend = None
        self._execution_count = 0
        self._backend_type = self._determine_backend_type()
        self._initialize_backend()
        self.logger.info("CircuitExecutor initialized with %s backend", self._backend_type.value)

    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance for execution tracking.
        
        Returns:
            Configured logger instance with appropriate handlers and formatters.
            Log level is set to INFO if verbose mode is enabled in configuration.
        """
        logger = logging.getLogger(f"CircuitExecutor.{id(self)}")
        
        if self.device_config.get('verbose', False):
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def _determine_backend_type(self) -> BackendType:
        """Determine the appropriate backend type from device configuration.
        
        Returns:
            BackendType enum value corresponding to the configured machine.

        Raises:
            ValueError: If machine_name is not recognized or supported
        """
        backend_name = self.device_config.get('machine_name', 'default')
        
        if backend_name == "default":
            return BackendType.LOCAL_SIMULATOR
        elif backend_name in self.TIANYAN_HARDWARE_BACKENDS:
            return BackendType.TIANYAN_HARDWARE
        elif backend_name in self.TIANYAN_SIMULATOR_BACKENDS:
            return BackendType.TIANYAN_SIMULATOR
        else:
            raise ValueError(f"Unknown or unsupported backend: {backend_name}")

    def _initialize_backend(self) -> None:
        """Initialize connection to the quantum computation backend.
        
        For local simulators, no connection is needed. For Tianyan platforms,
        this method establishes the API connection using provided credentials.

        Raises:
            ConnectionError: If remote backend connection fails
            ValueError: If required login credentials are missing
        """
        if self._backend_type == BackendType.LOCAL_SIMULATOR:
            self.logger.debug("Using local simulator - no backend connection needed")
            return
            
        login_key = self.device_config.get('login_key')
        if not login_key:
            raise ValueError("Login key required for Tianyan backend authentication")
            
        try:
            self.cqlib_backend = TianYanPlatform(
                login_key=login_key,
                machine_name=self.device_config['machine_name']
            )
            self.logger.info("Successfully connected to Tianyan backend: %s", 
                           self.device_config['machine_name'])
        except Exception as error:
            self.logger.error("Backend connection failed for %s: %s", 
                            self.device_config['machine_name'], error)
            raise ConnectionError(f"Backend connection failed: {error}") from error

    def execute_circuit(self, circuit: QuantumScript) -> Union[List, Any]:
        """Execute a quantum circuit and return measurement results.
        
        This is the main entry point for circuit execution. It handles the complete
        workflow including circuit validation, measurement processing, backend
        execution, and result formatting.

        Args:
            circuit: PennyLane QuantumScript object containing quantum operations
                and measurements to execute.

        Returns:
            Single measurement result if the circuit contains only one measurement,
            otherwise a list of results corresponding to each measurement in the circuit.

        Raises:
            ValueError: If circuit validation fails or backend doesn't support 
                       requested measurements
            NotImplementedError: For unsupported measurement types
            ConnectionError: If backend execution fails

        Example:
            >>> # Circuit with single measurement
            >>> result = executor.execute_circuit(tape)
            >>> print(f"Expectation value: {result}")
            
            >>> # Circuit with multiple measurements  
            >>> results = executor.execute_circuit(tape)
            >>> prob_result, sample_result = results
        """
        self._validate_circuit(circuit)
        self._execution_count += 1
        wire_labels = list(circuit.wires.labels)
        results = []
        
        for measurement in circuit.measurements:
            # Validate backend support before processing
            self._validate_measurement_support(measurement)
            
            cqlib_circuit, cqlib_qcis = self._convert_to_cqlib_format(circuit)

            raw_result = self._execute_on_backend(cqlib_circuit, cqlib_qcis)

            reordered_result = self._reorder_raw_result_by_wire_labels(raw_result, wire_labels)
            # Execute with measurement-specific handler
            
            result = self._execute_measurement(measurement, reordered_result)
            
            results.append(result)

        # Return single result for circuits with one measurement, list for multiple
        return results[0] if len(results) == 1 else results

    def _validate_circuit(self, circuit: QuantumScript) -> None:
        """Validate circuit constraints and configuration compatibility.
        
        Ensures that the circuit configuration is compatible with the selected
        backend and measurement types.

        Args:
            circuit: Quantum circuit to validate

        Raises:
            ValueError: If circuit contains state measurements with finite shots
                       or other incompatible configurations
        """
        # Check for state measurement with finite shots
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
        """Validate that the current backend supports the requested measurement type.
        
        Args:
            measurement: Measurement object to validate

        Raises:
            ValueError: If the backend doesn't support the measurement type
        """
        # Define measurement support matrix for each backend type
        supported_measurements = {
            BackendType.LOCAL_SIMULATOR: {
                qml.measurements.ProbabilityMP,   # Probability distributions
                qml.measurements.ExpectationMP,   # Expectation values
                qml.measurements.StateMP,         # Full statevector
                qml.measurements.SampleMP         # Measurement samples
            },
            BackendType.TIANYAN_SIMULATOR: {
                qml.measurements.ProbabilityMP,   # Probability distributions
                qml.measurements.SampleMP,        # Measurement samples
                qml.measurements.ExpectationMP,   # Expectation values
            },
            BackendType.TIANYAN_HARDWARE: {
                qml.measurements.ProbabilityMP,   # Probability distributions
                qml.measurements.SampleMP,        # Measurement samples
                qml.measurements.ExpectationMP,   # Expectation values
            }
        }
        
        measurement_type = type(measurement)
        if measurement_type not in supported_measurements[self._backend_type]:
            raise ValueError(
                f"Backend {self._backend_type.value} does not support "
                f"measurement type {measurement_type.__name__}. "
                f"Supported measurements: {[m.__name__ for m in supported_measurements[self._backend_type]]}"
            )

    def _convert_to_cqlib_format(self, circuit: QuantumScript) -> tuple[Any, str]:
        """Convert PennyLane circuit to CQLib-compatible format.
        
        Args:
            circuit: PennyLane QuantumScript to convert

        Returns:
            Tuple containing (cqlib_circuit_object, qcis_instruction_string)

        Raises:
            ValueError: If circuit conversion fails
        """
        try:
            qasm_string = circuit.to_openqasm()
            cqlib_circuit = qasm2.loads(qasm_string)
            return cqlib_circuit, cqlib_circuit.qcis
        except Exception as error:
            self.logger.error("Circuit conversion from PennyLane to CQLib format failed: %s", error)
            raise ValueError(f"Circuit conversion failed: {error}") from error

    def _execute_measurement(self, measurement: Any, raw_result: Dict[str, Any]) -> Any:
        """Execute circuit with measurement-specific processing.
        
        Dispatches to appropriate measurement handler based on measurement type
        using Python's singledispatchmethod for clean, extensible design.

        Args:
            measurement: Measurement object defining what to measure
            raw_result: Raw result dictionary from backend execution (already reordered)

        Returns:
            Processed measurement result in PennyLane-compatible format
        """
        return self._execute_measurement_impl(measurement, raw_result)

    @singledispatchmethod
    def _execute_measurement_impl(self, measurement: Any, raw_result: Dict[str, Any]) -> Any:
        """Base implementation for unsupported measurement types.
        
        This method is called when no specific handler is registered for
        the measurement type.

        Raises:
            NotImplementedError: Always raised for unregistered measurement types
        """
        raise NotImplementedError(
            f"Measurement type {type(measurement).__name__} is not supported. "
            f"Supported types: ProbabilityMP, ExpectationMP, StateMP, SampleMP"
        )

    @_execute_measurement_impl.register
    def _(self, measurement: qml.measurements.ProbabilityMP, raw_result: Dict[str, Any]) -> np.ndarray:
        """Execute probability measurement and return probability distribution.
        
        Probability measurements return an array where each element represents
        the probability of measuring the corresponding computational basis state.

        Args:
            measurement: Probability measurement object
            raw_result: Raw result dictionary from backend execution

        Returns:
            numpy.ndarray: Probability distribution over computational basis states.
            The array has length 2^n_qubits and sums to 1.0.

        Example:
            >>> # For a 2-qubit system, returns array of length 4
            >>> probs = executor.execute_circuit(tape_with_prob_measurement)
            >>> print(f"Probability of |00>: {probs[0]}")
        """
        probabilities = self._extract_probabilities(raw_result)
        return self._format_probabilities(probabilities)

    @_execute_measurement_impl.register  
    def _(self, measurement: qml.measurements.ExpectationMP, raw_result: Dict[str, Any]) -> float:
        """Execute expectation value measurement for Pauli-Z observables.
        
        Computes the expectation value of Pauli-Z observables and Pauli-Z strings
        directly from probability distribution results. This method assumes the
        circuit has already been transformed to the Z-basis measurement.

        Args:
            measurement: Expectation measurement object containing a Pauli-Z observable
            raw_result: Raw result dictionary containing 'probabilities' key with
                    a dictionary mapping bitstrings to probabilities

        Returns:
            float: Expectation value of the Pauli-Z observable

        Note:
            - Only supports Pauli-Z observables and tensor products of Pauli-Z
            - Assumes basis transformation has been applied prior to measurement
            - Works with probability distributions from both simulators and hardware
            - For non-Z observables, use basis transformation in the circuit

        Example:
            >>> # Expectation of Z(0) ⊗ Z(1) ⊗ Z(2)
            >>> obs = qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2)
            >>> expval = executor.execute_circuit(tape_with_expval_measurement)
            >>> print(f"<Z0⊗Z1⊗Z2> = {expval}")

        Raises:
            ValueError: If raw_result doesn't contain probability distribution
            KeyError: If required keys are missing in raw_result
        """
        # Validate input
        if 'probabilities' not in raw_result:
            raise ValueError("raw_result must contain 'probabilities' key")
        
        probabilities = raw_result['probabilities']
        if not isinstance(probabilities, dict):
            raise ValueError("probabilities must be a dictionary")
        
        # Extract qubit indices from observable
        pauli_indices = list(measurement.obs.wires.labels)
        
        expectation = 0.0
        
        for bitstring, prob in probabilities.items():
            # Calculate eigenvalue for this basis state
            eigenvalue = 1
            for qubit_idx in pauli_indices:
                if qubit_idx < len(bitstring):
                    # For Pauli-Z: |0⟩ → +1, |1⟩ → -1
                    if bitstring[qubit_idx] == '1':
                        eigenvalue *= -1
            
            # Expectation = Σ (probability × eigenvalue)
            expectation += prob * eigenvalue
        
        return expectation
        
    @_execute_measurement_impl.register
    def _(self, measurement: qml.measurements.StateMP, raw_result: Dict[str, Any]) -> np.ndarray:
        """Execute statevector measurement and return the full quantum state.
        
        Returns the complete statevector of the quantum system. This measurement
        is only supported on local statevector simulators.

        Args:
            measurement: State measurement object
            raw_result: Raw result dictionary from backend execution

        Returns:
            numpy.ndarray: Complex-valued statevector of length 2^n_qubits

        Raises:
            ValueError: If attempted on non-local simulator backend

        Example:
            >>> statevector = executor.execute_circuit(tape_with_state_measurement)
            >>> print(f"Statevector shape: {statevector.shape}")
        """
        if self._backend_type != BackendType.LOCAL_SIMULATOR:
            raise ValueError(
                "Statevector measurement is only supported on local simulators. "
                f"Current backend: {self._backend_type.value}"
            )
            
        statevector = raw_result.get('statevector')
        if statevector is None:
            raise ValueError("Statevector not found in execution results")
        return statevector

    @_execute_measurement_impl.register
    def _(self, measurement: qml.measurements.SampleMP, raw_result: Dict[str, Any]) -> np.ndarray:
        """Execute sampling measurement and return measurement samples.
        
        Returns raw measurement samples from multiple circuit executions.
        Each sample is a bitstring representing the measurement outcome.

        Args:
            measurement: Sample measurement object
            raw_result: Raw result dictionary from backend execution

        Returns:
            numpy.ndarray: Array of shape (n_shots, n_qubits) where each row
            is a measurement outcome and each column is a qubit measurement result.

        Example:
            >>> samples = executor.execute_circuit(tape_with_sample_measurement)
            >>> print(f"Samples shape: {samples.shape}")
            >>> print(f"First measurement: {samples[0]}")
        """
        samples = raw_result.get('samples')
        if samples is None:
            raise ValueError("No measurement samples found in execution results")
        return samples
    
    def _execute_on_backend(self, cqlib_circuit: Any, cqlib_qcis: str) -> Dict[str, Any]:
        """Execute circuit on the appropriate backend and return raw results.
        
        Args:
            cqlib_circuit: Circuit in CQLib object format
            cqlib_qcis: Circuit in QCIS instruction format

        Returns:
            Dictionary containing raw results from backend execution

        Raises:
            ConnectionError: If backend execution fails
        """
        backend_handlers = {
            BackendType.LOCAL_SIMULATOR: self._execute_local_simulator,
            BackendType.TIANYAN_SIMULATOR: self._execute_tianyan_simulator,
            BackendType.TIANYAN_HARDWARE: self._execute_tianyan_hardware
        }
        
        handler = backend_handlers[self._backend_type]
        return handler(cqlib_circuit, cqlib_qcis)

    def _execute_local_simulator(self, cqlib_circuit: Any, cqlib_qcis: str) -> Dict[str, Any]:
        """Execute circuit on local statevector simulator.
        
        Returns comprehensive results including probabilities, samples, and
        statevector for local simulation.

        Returns:
            Dictionary with keys: 'probabilities', 'samples', 'statevector'
        """
        simulator = StatevectorSimulator(cqlib_circuit)
        nwe = dict(reversed(simulator.probs().items()))
        
        reversed_statevector = {key[::-1]: value for key, value in simulator.statevector().items()}
        return {
            'probabilities': simulator.probs(),
            'samples': simulator.sample(is_raw_data=True),
            'statevector': reversed_statevector
        }

    def _execute_tianyan_simulator(self, cqlib_circuit: Any, cqlib_qcis: str) -> Dict[str, Any]:
        """Execute circuit on Tianyan cloud simulator.
        
        Returns probability distributions and measurement samples from
        Tianyan's cloud-based simulators.

        Returns:
            Dictionary with keys: 'probabilities', 'samples'
        """
        query_id = self.cqlib_backend.submit_experiment(
            cqlib_qcis, 
            num_shots=self.device_config.get('shots')
        )
        raw_result = self.cqlib_backend.query_experiment(query_id)[0]
        sample_res = np.array(raw_result['resultStatus'][1:])
        return {
            'probabilities': json.loads(raw_result['probability']),
            'samples': sample_res
        }

    def _execute_tianyan_hardware(self, cqlib_circuit: Any, cqlib_qcis: str) -> Dict[str, Any]:
        """Execute circuit on Tianyan quantum hardware.
        
        Submits circuit to physical quantum hardware and returns measurement
        samples. Hardware execution includes readout calibration and error
        mitigation where available.

        Returns:
            Dictionary with key: 'samples'

        Note:
            Hardware execution may involve queueing and longer execution times
        """
        from .ext_mapping import HardwareMapper
        mapper = self.device_config.get('mapping', None)
        if mapper:
            MAP = HardwareMapper(mapper)
            compiled_circuit = MAP.map_qcis_code(cqlib_circuit.qcis)

        else:
            compiled_circuit = transpile_qcis(cqlib_qcis, self.cqlib_backend)[0]
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

    def _reorder_raw_result_by_wire_labels(self, raw_result: Dict[str, Any], 
                                            wire_labels: List[int]) -> Dict[str, Any]:
            """Reorder raw results to match PennyLane's wire label ordering.
            
            PennyLane may reorder qubits during compilation, and circuit.wires.labels
            indicates the final classical bit ordering. This method reorders raw results
            before they are processed by measurement-specific handlers.
            
            Args:
                raw_result: The raw result dictionary from backend execution
                wire_labels: List of wire labels from circuit.wires.labels, e.g., [0, 2, 1]
                
            Returns:
                Reordered raw result dictionary matching the wire_labels ordering
            """
            
            reordered_result = raw_result.copy()
            
            # Reorder probabilities if present
            if 'probabilities' in raw_result and raw_result['probabilities']:
                reordered_result['probabilities'] = self._reorder_probability_dict(
                    raw_result['probabilities'], wire_labels
                )
            
            # Reorder samples if present
            if 'samples' in raw_result and raw_result['samples'] is not None:
                sample = self._extract_samples(raw_result)
                reordered_result['samples'] = self._reorder_sample_matrix(
                    sample, wire_labels
                )
            
            # Reorder statevector if present
            if 'statevector' in raw_result and raw_result['statevector'] is not None:
                statevector = raw_result['statevector']
                reordered_result['statevector'] = self._reorder_statevector(
                    statevector, wire_labels
                )
            
            return reordered_result

    def _reorder_probability_dict(self, probabilities: Dict[str, float],
                              wire_labels: List[int]) -> Dict[str, float]:
        """Reorder probability dictionary based on wire labels mapping."""
        reordered_probabilities = {}
        
        for bitstring, probability in probabilities.items():
            # Convert to list and reverse for little-endian to big-endian
            bits = list(bitstring)[::-1]
            
            # Create new bit array and apply wire label mapping
            reordered_bits = ['0'] * len(wire_labels)
            for new_pos, original_pos in enumerate(wire_labels):
                reordered_bits[original_pos] = bits[new_pos]
            
            # Convert back to string format
            reordered_bitstring = ''.join(reordered_bits[::-1])
            reordered_probabilities[reordered_bitstring] = probability
        
        return reordered_probabilities

    def _reorder_sample_matrix(self, samples: np.ndarray, 
                                wire_labels: List[int]) -> np.ndarray:
            """Reorder sample matrix based on wire labels.
            
            Args:
                samples: Sample matrix of shape (n_shots, n_qubits)
                wire_labels: Desired wire ordering, e.g., [0, 2, 1]
                
            Returns:
                Reordered sample matrix
            """
            n_qubits = len(wire_labels)
            
            # Create the permutation to go from current order to desired order
            current_to_desired = [wire_labels.index(i) for i in range(n_qubits)]
            
            # Reorder columns
            return samples[:, current_to_desired]
    
    def _reorder_statevector(self, statevector: Dict[str, complex],
                       wire_labels: List[int]) -> Dict[str, complex]:
        """Reorder statevector dictionary based on wire labels.
        
        Args:
            statevector: Dictionary mapping bitstrings to complex amplitudes
            wire_labels: Desired wire ordering, e.g., [0, 2, 1]
            
        Returns:
            Reordered statevector dictionary
        """
        n_qubits = len(wire_labels)
        
        # Create the permutation to go from current order to desired order
        # For example, if wire_labels = [0, 2, 1], then:
        # current_to_desired = [0, 2, 1] means:
        # - bit 0 stays at position 0
        # - bit 1 goes to position 2  
        # - bit 2 goes to position 1
        current_to_desired = [wire_labels.index(i) for i in range(n_qubits)]
        
        reordered_statevector = {}
        
        for bitstring, amplitude in statevector.items():
            # Convert bitstring to list of bits
            bits = list(bitstring)
            # Reorder bits according to the permutation
            reordered_bits = [bits[current_to_desired[i]] for i in range(n_qubits)]
            reordered_bitstring = ''.join(reordered_bits)
            
            # Store with reordered bitstring
            reordered_statevector[reordered_bitstring] = amplitude
            
        return reordered_statevector  

    def _extract_probabilities(self, raw_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract and process probability distribution from raw results.
        
        Handles endianness conversion to ensure consistent little-endian
        format across all backends.

        Args:
            raw_result: Raw result dictionary from backend execution

        Returns:
            Probability dictionary mapping bitstrings to probabilities
        """
        probabilities = raw_result.get('probabilities', {})
        
        # Convert to little-endian format for consistency
        if probabilities and isinstance(probabilities, dict):
            return {key[::-1]: value for key, value in probabilities.items()}
        
        return probabilities

    def _extract_samples(self, raw_result: Dict[str, Any]) -> Any:
        """Extract and format samples from raw results.
        
        Converts local simulator samples to PennyLane-compatible format
        while preserving Tianyan backend sample formats.

        Args:
            raw_result: Raw result dictionary from backend execution

        Returns:
            Formatted samples appropriate for the backend type
        """
        samples = raw_result.get('samples')
        
        # Convert local simulator samples to standard format
        if samples is not None and self._backend_type == BackendType.LOCAL_SIMULATOR:
            return samples_to_pennylane_format(samples, self.device_config['wires'])
        
        return samples

    def _format_probabilities(self, probabilities: Dict[str, float]) -> np.ndarray:
        """Convert probability dictionary to PennyLane array format.
        
        Args:
            probabilities: Dictionary mapping bitstrings to probabilities

        Returns:
            numpy.ndarray: Probability array indexed by computational basis states

        Raises:
            ValueError: If probability dictionary is empty or invalid
        """
        if not probabilities:
            raise ValueError("No probability distribution found in execution results")
            
        num_qubits = len(next(iter(probabilities.keys())))
        prob_array = np.zeros(2 ** num_qubits)
        
        for bitstring, prob in probabilities.items():
            index = int(bitstring, 2)  # Convert binary string to integer index
            prob_array[index] = prob
            
        return prob_array

    def _format_samples(self, samples: Any, measurement: qml.measurements.SampleMP) -> np.ndarray:
        """Format samples for PennyLane compatibility.
        
        Args:
            samples: Raw samples from backend execution
            measurement: Sample measurement object for context

        Returns:
            Formatted samples array

        Raises:
            ValueError: If no samples are found in results
        """
        if samples is None:
            raise ValueError("No measurement samples found in execution results")
        return samples

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics and performance metrics.
        
        Returns:
            Dictionary containing execution count, backend information,
            and configuration details.

        Example:
            >>> stats = executor.get_execution_stats()
            >>> print(f"Total executions: {stats['execution_count']}")
            >>> print(f"Backend type: {stats['backend_type']}")
        """
        return {
            "execution_count": self._execution_count,
            "backend_type": self._backend_type.value,
            "wires": self.device_config.get('wires'),
            "shots": self.device_config.get('shots'),
            "machine_name": self.device_config.get('machine_name')
        }


def decimal_to_binary_array(
    decimal_value: int, 
    num_bits: int, 
    little_endian: bool = True
) -> np.ndarray:
    """Convert decimal integer to binary array representation.
    
    Args:
        decimal_value: Integer value to convert to binary
        num_bits: Number of bits in the binary representation
        little_endian: If True, least significant bit is at index 0.
                      If False, most significant bit is at index 0.

    Returns:
        numpy.ndarray: Binary array of length num_bits containing 0s and 1s

    Example:
        >>> decimal_to_binary_array(5, 4, little_endian=True)
        array([1, 0, 1, 0])  # 5 = 1*2^0 + 0*2^1 + 1*2^2 + 0*2^3
        >>> decimal_to_binary_array(5, 4, little_endian=False)  
        array([0, 1, 0, 1])  # 5 = 0*2^3 + 1*2^2 + 0*2^1 + 1*2^0
    """
    binary_string = np.binary_repr(int(decimal_value), width=num_bits)
    bits = np.array([int(bit) for bit in binary_string])
    return bits[::-1] if little_endian else bits


def samples_to_pennylane_format(
    samples: Union[List[int], np.ndarray],
    num_qubits: Optional[int] = None,
    measured_qubits: Optional[List[int]] = None,
    little_endian: bool = True
) -> np.ndarray:
    """Convert decimal samples to PennyLane-compatible binary matrix.
    
    Args:
        samples: Array of decimal integers representing measurement outcomes
        num_qubits: Total number of qubits in the system
        measured_qubits: Specific qubits that were measured
        little_endian: Endianness convention for bit ordering

    Returns:
        numpy.ndarray: Binary matrix of shape (n_shots, n_bits) where
        each row is a measurement outcome and each column is a qubit result.

    Raises:
        ValueError: If number of bits cannot be determined from inputs

    Example:
        >>> samples = [1, 3, 2]  # Decimal measurement outcomes
        >>> samples_to_pennylane_format(samples, num_qubits=2)
        array([[1, 0],  # 1 in binary (little-endian)
               [1, 1],  # 3 in binary  
               [0, 1]]) # 2 in binary
    """
    samples_array = np.asarray(samples)
    
    # Determine required number of bits
    if measured_qubits is not None:
        num_bits = len(measured_qubits)
    elif num_qubits is not None:
        num_bits = num_qubits
    elif len(samples_array) == 0:
        raise ValueError("Cannot determine number of bits from empty samples array")
    else:
        max_value = np.max(samples_array)
        num_bits = int(np.ceil(np.log2(max_value + 1))) if max_value > 0 else 1

    # Convert each sample to binary representation
    n_shots = len(samples_array)
    binary_matrix = np.zeros((n_shots, num_bits), dtype=int)
    
    for i, sample in enumerate(samples_array):
        binary_matrix[i] = decimal_to_binary_array(sample, num_bits, little_endian)
    
    return binary_matrix


def switch_endianness(
    binary_data: Union[List[int], np.ndarray, List[List[int]]]
) -> np.ndarray:
    """Reverse the bit order (endianness) of binary data.
    
    Args:
        binary_data: Binary data to convert. Can be 1D array (single measurement)
                    or 2D array (multiple measurements, each row is a bitstring)

    Returns:
        numpy.ndarray: Binary data with bit order reversed along the last axis

    Example:
        >>> switch_endianness([1, 0, 1, 0])
        array([0, 1, 0, 1])  # Big-endian to little-endian
        >>> switch_endianness([[1, 0], [0, 1]])
        array([[0, 1],       # Each row reversed independently
               [1, 0]])
    """
    data_array = np.asarray(binary_data)
    return data_array[..., ::-1]  # Reverse along the last axis (bit dimension)