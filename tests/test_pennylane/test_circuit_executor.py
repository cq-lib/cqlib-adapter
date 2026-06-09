# This code is part of cqlib.
#
# Copyright (C) 2025-2026 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Comprehensive Test Suite for Quantum Circuit Executor.

This module provides extensive unit testing for the CircuitExecutor class,
covering backend initialization, circuit execution, measurement processing,
result formatting, and error handling protocols.
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pennylane as qml
import pytest
from pennylane.tape import QuantumScript

from cqlib_adapter.pennylane_ext.circuit_executor import (
    BackendType,
    CircuitExecutor,
    decimal_to_binary_array,
    samples_to_pennylane_format,
)


@pytest.fixture(autouse=True)
def mock_cqlib_device_backends():
    """Injects mocked backend lists into CQLibDevice for isolated testing.

    Overrides the remote API call dependency by manually populating
    the supported hardware and simulator backend caches. Ensures the
    original state is cleanly restored after test execution to prevent
    state leakage between tests.
    """
    from cqlib_adapter.pennylane_ext.device import CQLibDevice
    
    # Store the original state to guarantee test isolation
    orig_hw = getattr(CQLibDevice, 'TIANYAN_HARDWARE_BACKENDS', [])
    orig_sim = getattr(CQLibDevice, 'TIANYAN_SIMULATOR_BACKENDS', [])
    
    # Inject mock configuration for backend validation
    CQLibDevice.TIANYAN_HARDWARE_BACKENDS = ['tianyan24', 'tianyan504']
    CQLibDevice.TIANYAN_SIMULATOR_BACKENDS = ['tianyan_sw', 'tianyan_s']
    
    yield  # Suspend execution to run the test
    
    # Restore the original state during teardown
    CQLibDevice.TIANYAN_HARDWARE_BACKENDS = orig_hw
    CQLibDevice.TIANYAN_SIMULATOR_BACKENDS = orig_sim


class TestCircuitExecutorInitialization:
    """Test suite for CircuitExecutor initialization and configuration mapping."""
    
    def test_local_simulator_initialization(self):
        """Tests if the executor initializes correctly with the default local simulator."""
        # Arrange
        config = {
            'machine_name': 'default',
            'shots': 1000,
            'wires': 2,
            'verbose': True
        }
        
        # Act
        executor = CircuitExecutor(config)
        
        # Assert
        assert executor._backend_type == BackendType.LOCAL_SIMULATOR
        assert executor.device_config == config
        assert executor.cqlib_backend is None
        assert executor._execution_count == 0

    @patch('cqlib_adapter.pennylane_ext.device.CQLibDevice.get_available_backends')
    @patch('cqlib_adapter.pennylane_ext.circuit_executor.TianYanPlatform')
    def test_tianyan_simulator_initialization(self, mock_platform, mock_get_backends):
        """Tests initialization when a Tianyan simulator backend is requested."""
        # Arrange
        config = {
            'machine_name': 'tianyan_s',
            'login_key': 'test_key',
            'shots': 1000,
            'wires': 5,
            'verbose': False
        }
        mock_instance = Mock()
        mock_platform.return_value = mock_instance
        
        # Act
        executor = CircuitExecutor(config)
        
        # Assert
        assert executor._backend_type == BackendType.TIANYAN_SIMULATOR
        mock_platform.assert_called_once_with(
            login_key='test_key',
            machine_name='tianyan_s'
        )
        mock_get_backends.assert_called_once_with(token='test_key')
            
    @patch('cqlib_adapter.pennylane_ext.device.CQLibDevice.get_available_backends')
    @patch('cqlib_adapter.pennylane_ext.circuit_executor.TianYanPlatform')
    def test_tianyan_hardware_initialization(self, mock_platform, mock_get_backends):
        """Tests initialization when a Tianyan hardware backend is requested."""
        # Arrange
        config = {
            'machine_name': 'tianyan24',
            'login_key': 'test_key',
            'shots': 1000,
            'wires': 24,
            'verbose': True
        }
        mock_instance = Mock()
        mock_platform.return_value = mock_instance
        
        # Act
        executor = CircuitExecutor(config)
        
        # Assert
        assert executor._backend_type == BackendType.TIANYAN_HARDWARE
        mock_platform.assert_called_once_with(
            login_key='test_key',
            machine_name='tianyan24'
        )
            
    def test_invalid_backend_initialization(self):
        """Tests that a ValueError is raised when an unsupported backend is provided."""
        config = {
            'machine_name': 'invalid_backend',
            'shots': 1000,
            'wires': 2
        }
        
        with pytest.raises(ValueError, match="Login key required"):
            CircuitExecutor(config)
            
    def test_missing_login_key_for_tianyan(self):
        """Tests that a ValueError is raised when a remote backend lacks a login key."""
        config = {
            'machine_name': 'tianyan_s',
            'shots': 1000,
            'wires': 5
        }
        
        with pytest.raises(ValueError, match="Login key required"):
            CircuitExecutor(config)


class TestBackendTypeDetermination:
    """Test suite validating the mapping from machine names to BackendType enums."""
    
    @pytest.mark.parametrize("machine_name,expected_type", [
        ('default', BackendType.LOCAL_SIMULATOR),
        ('tianyan24', BackendType.TIANYAN_HARDWARE),
        ('tianyan504', BackendType.TIANYAN_HARDWARE),
        ('tianyan_sw', BackendType.TIANYAN_SIMULATOR),
        ('tianyan_s', BackendType.TIANYAN_SIMULATOR),
    ])
    @patch('cqlib_adapter.pennylane_ext.device.CQLibDevice.get_available_backends')
    @patch('cqlib_adapter.pennylane_ext.circuit_executor.TianYanPlatform')
    def test_backend_type_mapping(self, mock_platform, mock_get_backends, machine_name, expected_type):
        """Tests that all supported machine names map to their correct BackendType."""
        config = {'machine_name': machine_name, 'wires': 2}
        
        if expected_type != BackendType.LOCAL_SIMULATOR:
            config['login_key'] = 'test_key'
            
        executor = CircuitExecutor(config)
        assert executor._backend_type == expected_type


class TestCircuitValidation:
    """Test suite for validating circuit configurations prior to execution."""
    
    def test_state_measurement_with_finite_shots(self):
        """Ensures state vector measurements fail if finite shots are configured."""
        # Arrange
        config = {
            'machine_name': 'default',
            'shots': 1000,
            'wires': 2
        }
        executor = CircuitExecutor(config)
        ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
        measurements = [qml.state()]
        circuit = QuantumScript(ops, measurements)
        
        # Act & Assert
        with pytest.raises(ValueError, match="State measurement requires shots=None"):
            executor._validate_circuit(circuit)
            
    def test_state_measurement_with_infinite_shots(self):
        """Ensures state vector measurements pass validation when shots are None."""
        # Arrange
        config = {
            'machine_name': 'default',
            'shots': None,
            'wires': 2
        }
        executor = CircuitExecutor(config)
        ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
        measurements = [qml.state()]
        circuit = QuantumScript(ops, measurements)
        
        # Act & Assert
        executor._validate_circuit(circuit)  # Should not raise an exception


class TestMeasurementSupportValidation:
    """Test suite validating backend-specific measurement capabilities."""
    
    @pytest.mark.parametrize("backend_type,measurement_type,should_support", [
        (BackendType.LOCAL_SIMULATOR, qml.measurements.ProbabilityMP, True),
        (BackendType.LOCAL_SIMULATOR, qml.measurements.StateMP, True),
        (BackendType.TIANYAN_SIMULATOR, qml.measurements.ProbabilityMP, True),
        (BackendType.TIANYAN_SIMULATOR, qml.measurements.StateMP, False),
        (BackendType.TIANYAN_HARDWARE, qml.measurements.SampleMP, True),
        (BackendType.TIANYAN_HARDWARE, qml.measurements.StateMP, False),
    ])
    def test_measurement_support(self, backend_type, measurement_type, should_support):
        """Verifies if the requested backend permits the provided measurement type."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        executor._backend_type = backend_type
        
        measurement = measurement_type()
        
        if should_support:
            executor._validate_measurement_support(measurement)
        else:
            with pytest.raises(ValueError, match="does not support"):
                executor._validate_measurement_support(measurement)


class TestCircuitExecution:
    """Test suite for the primary circuit execution pipeline."""
    
    def test_single_measurement_execution(self):
        """Tests the execution flow for a circuit returning a single measurement."""
        # Arrange
        config = {
            'machine_name': 'default',
            'shots': None,
            'wires': 2
        }
        executor = CircuitExecutor(config)
        
        # Act
        with patch.object(executor, '_convert_to_cqlib_format') as mock_convert, \
             patch.object(executor, '_execute_on_backend') as mock_execute, \
             patch.object(executor, '_execute_measurement') as mock_measure:
            
            mock_convert.return_value = (Mock(), "test_qcis")
            mock_execute.return_value = {'probabilities': {'00': 0.5, '11': 0.5}}
            mock_measure.return_value = np.array([0.5, 0.0, 0.0, 0.5])
            
            ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
            measurements = [qml.probs()]
            circuit = QuantumScript(ops, measurements)
            
            result = executor.execute_circuit(circuit)
            
            # Assert
            assert isinstance(result, np.ndarray)
            assert mock_convert.called
            assert mock_execute.called
            assert mock_measure.called
            
    def test_multiple_measurements_execution(self):
        """Tests the execution flow handling circuits with multiple measurements."""
        # Arrange
        config = {
            'machine_name': 'default', 
            'shots': None,
            'wires': 2
        }
        executor = CircuitExecutor(config)
        
        # Act
        with patch.object(executor, '_convert_to_cqlib_format') as mock_convert, \
             patch.object(executor, '_execute_on_backend') as mock_execute, \
             patch.object(executor, '_execute_measurement') as mock_measure:
            
            mock_convert.return_value = (Mock(), "test_qcis")
            mock_execute.return_value = {'probabilities': {'00': 1.0}}
            mock_measure.side_effect = [
                np.array([1.0, 0.0, 0.0, 0.0]),
                1.0
            ]
            
            ops = [qml.Identity(0)]
            measurements = [qml.probs(), qml.expval(qml.PauliZ(0))]
            circuit = QuantumScript(ops, measurements)
            
            results = executor.execute_circuit(circuit)
            
            # Assert
            assert isinstance(results, list)
            assert len(results) == 2
            assert mock_measure.call_count == 2


class TestMeasurementProcessing:
    """Test suite verifying result parsing based on specific measurement types."""
    
    def test_probability_measurement_processing(self):
        """Tests extraction and formatting of probability measurements."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        
        raw_result = {
            'probabilities': {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25}
        }
        measurement = qml.measurements.ProbabilityMP()
        
        result = executor._execute_measurement_impl(measurement, raw_result)
        
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_equal(result, expected)
        
    def test_expectation_measurement_processing(self):
        """Tests parsing logic for expectation value computations."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        
        raw_result = {
            'probabilities': {'00': 0.5, '11': 0.5}
        }
        measurement = qml.measurements.ExpectationMP(qml.PauliZ(0) @ qml.PauliZ(1))
        
        result = executor._execute_measurement_impl(measurement, raw_result)
        
        assert result == 1.0
        
        
class TestBackendExecution:
    """Test suite validating direct invocations against specific backend targets."""
    
    def test_local_simulator_execution(self):
        """Tests direct payload submission to the local statevector simulator."""
        # Arrange
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        mock_circuit = Mock()
        mock_simulator = Mock()
        
        # Act & Assert
        with patch('cqlib_adapter.pennylane_ext.circuit_executor.StatevectorSimulator') as mock_sim_class:
            mock_sim_class.return_value = mock_simulator
            mock_simulator.statevector.return_value = {'00': 1.0}
            mock_simulator.probs.return_value = {'00': 1.0}
            mock_simulator.sample.return_value = np.array([[0, 0]])
            
            result = executor._execute_local_simulator(mock_circuit, "test_qcis")
            
            assert 'probabilities' in result
            assert 'samples' in result
            assert 'statevector' in result
            mock_sim_class.assert_called_once_with(mock_circuit)
            
    @patch('cqlib_adapter.pennylane_ext.device.CQLibDevice.get_available_backends')
    @patch('cqlib_adapter.pennylane_ext.circuit_executor.TianYanPlatform')
    def test_tianyan_simulator_execution(self, mock_platform_class, mock_get_backends):
        """Tests the submission and retrieval logic via the TianYan cloud platform."""
        # Arrange
        config = {
            'machine_name': 'tianyan_s',
            'login_key': 'test_key',
            'shots': 1000,
            'wires': 2
        }
        mock_platform = Mock()
        mock_platform_class.return_value = mock_platform
        executor = CircuitExecutor(config)
        
        mock_platform.submit_experiment.return_value = 'test_query_id'
        mock_platform.query_experiment.return_value = [{
            'resultStatus': 'S[0.5,0.5]',
            'probability': '{"00":0.5,"11":0.5}'
        }]
        
        mock_circuit = Mock()
        mock_circuit.qcis = "test_qcis"
        
        # Act
        result = executor._execute_tianyan_simulator(mock_circuit, "test_qcis")
        
        # Assert
        assert 'probabilities' in result
        assert 'samples' in result
        mock_platform.submit_experiment.assert_called_once_with(
            "test_qcis", num_shots=1000
        )

        

class TestUtilityFunctions:
    """Test suite for binary conversion and formatting data utilities."""
    
    @pytest.mark.parametrize("decimal,bits,little_endian,expected", [
        (5, 4, True, [1, 0, 1, 0]),
        (5, 4, False, [0, 1, 0, 1]),
        (0, 3, True, [0, 0, 0]),
        (7, 3, True, [1, 1, 1]),
    ])
    def test_decimal_to_binary_array(self, decimal, bits, little_endian, expected):
        """Tests endianness handling during decimal-to-binary transformation."""
        result = decimal_to_binary_array(decimal, bits, little_endian)
        expected_array = np.array(expected)
        np.testing.assert_array_equal(result, expected_array)
        
    def test_samples_to_pennylane_format(self):
        """Tests the restructuring of raw integers into PennyLane sample matrices."""
        samples = [1, 2, 3]
        num_qubits = 2
        
        result = samples_to_pennylane_format(samples, num_qubits)
        
        expected = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
        ])
        np.testing.assert_array_equal(result, expected)


class TestErrorHandling:
    """Test suite simulating fault tolerance and exception propagation."""
    
    @patch('cqlib_adapter.pennylane_ext.device.CQLibDevice.get_available_backends')
    def test_backend_connection_failure(self, mock_get_backends):
        """Tests graceful failure on critical external API timeouts."""
        config = {
            'machine_name': 'tianyan_s',
            'login_key': 'test_key',
            'wires': 2
        }
        
        mock_get_backends.side_effect = Exception("Connection failed")
        
        with pytest.raises(ConnectionError, match="Could not connect to Tianyan API"):
            CircuitExecutor(config)
                
class TestIntegration:
    """Integration test suite executing components from initialization to measurement."""        
    def test_logging_setup(self):
        """Tests contextually accurate configuration of runtime loggers."""
        config = {
            'machine_name': 'default',
            'shots': 1000,
            'wires': 2,
            'verbose': True
        }
        
        executor = CircuitExecutor(config)
        
        assert executor.logger is not None
        assert executor.logger.level == logging.INFO
        
        config['verbose'] = False
        executor = CircuitExecutor(config)
        assert executor.logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])