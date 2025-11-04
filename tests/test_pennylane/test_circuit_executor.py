"""
Comprehensive Test Suite for Quantum Circuit Executor.

This module provides extensive testing for the CircuitExecutor class,
covering all major functionality including backend initialization,
circuit execution, measurement processing, and error handling.
"""

import pytest
import numpy as np
import pennylane as qml
from unittest.mock import Mock, patch, MagicMock
import logging
from pennylane.tape import QuantumScript

from cqlib_adapter.pennylane_ext.circuit_executor import (
    CircuitExecutor, 
    BackendType, 
    decimal_to_binary_array,
    samples_to_pennylane_format,
    switch_endianness
)


class TestCircuitExecutorInitialization:
    """Test suite for CircuitExecutor initialization and configuration."""
    
    def test_local_simulator_initialization(self):
        """Test initialization with local simulator backend."""
        config = {
            'machine_name': 'default',
            'shots': 1000,
            'wires': 2,
            'verbose': True
        }
        
        executor = CircuitExecutor(config)
        
        assert executor._backend_type == BackendType.LOCAL_SIMULATOR
        assert executor.device_config == config
        assert executor.cqlib_backend is None
        assert executor._execution_count == 0
        
    def test_tianyan_simulator_initialization(self):
        """Test initialization with Tianyan simulator backend."""
        config = {
            'machine_name': 'tianyan_s',
            'login_key': 'test_key',
            'shots': 1000,
            'wires': 5,
            'verbose': False
        }
        
        with patch('cqlib_adapter.pennylane_ext.circuit_executor.TianYanPlatform') as mock_platform:
            mock_instance = Mock()
            mock_platform.return_value = mock_instance
            
            executor = CircuitExecutor(config)
            
            assert executor._backend_type == BackendType.TIANYAN_SIMULATOR
            mock_platform.assert_called_once_with(
                login_key='test_key',
                machine_name='tianyan_s'
            )
            
    def test_tianyan_hardware_initialization(self):
        """Test initialization with Tianyan hardware backend."""
        config = {
            'machine_name': 'tianyan24',
            'login_key': 'test_key',
            'shots': 1000,
            'wires': 24,
            'verbose': True
        }
        
        with patch('cqlib_adapter.pennylane_ext.circuit_executor.TianYanPlatform') as mock_platform:
            mock_instance = Mock()
            mock_platform.return_value = mock_instance
            
            executor = CircuitExecutor(config)
            
            assert executor._backend_type == BackendType.TIANYAN_HARDWARE
            mock_platform.assert_called_once_with(
                login_key='test_key',
                machine_name='tianyan24'
            )
            
    def test_invalid_backend_initialization(self):
        """Test initialization with invalid backend name."""
        config = {
            'machine_name': 'invalid_backend',
            'shots': 1000,
            'wires': 2
        }
        
        with pytest.raises(ValueError, match="Unknown or unsupported backend"):
            CircuitExecutor(config)
            
    def test_missing_login_key_for_tianyan(self):
        """Test initialization without login key for Tianyan backends."""
        config = {
            'machine_name': 'tianyan_s',
            'shots': 1000,
            'wires': 5
        }
        
        with pytest.raises(ValueError, match="Login key required"):
            CircuitExecutor(config)


class TestBackendTypeDetermination:
    """Test backend type determination logic."""
    
    @pytest.mark.parametrize("machine_name,expected_type", [
        ('default', BackendType.LOCAL_SIMULATOR),
        ('tianyan24', BackendType.TIANYAN_HARDWARE),
        ('tianyan504', BackendType.TIANYAN_HARDWARE),
        ('tianyan_sw', BackendType.TIANYAN_SIMULATOR),
        ('tianyan_s', BackendType.TIANYAN_SIMULATOR),
    ])
    def test_backend_type_mapping(self, machine_name, expected_type):
        """Test all supported backend type mappings."""
        config = {'machine_name': machine_name, 'wires': 2}
        
        if expected_type != BackendType.LOCAL_SIMULATOR:
            config['login_key'] = 'test_key'
            with patch('cqlib_adapter.pennylane_ext.circuit_executor.TianYanPlatform'):
                executor = CircuitExecutor(config)
        else:
            executor = CircuitExecutor(config)
            
        assert executor._backend_type == expected_type


class TestCircuitValidation:
    """Test circuit validation functionality."""
    
    def test_state_measurement_with_finite_shots(self):
        """Test validation fails for state measurement with finite shots."""
        config = {
            'machine_name': 'default',
            'shots': 1000,
            'wires': 2
        }
        
        executor = CircuitExecutor(config)
        
        # Create a circuit with state measurement
        ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
        measurements = [qml.state()]
        circuit = QuantumScript(ops, measurements)
        
        with pytest.raises(ValueError, match="State measurement requires shots=None"):
            executor._validate_circuit(circuit)
            
    def test_state_measurement_with_infinite_shots(self):
        """Test validation passes for state measurement with shots=None."""
        config = {
            'machine_name': 'default',
            'shots': None,
            'wires': 2
        }
        
        executor = CircuitExecutor(config)
        
        ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
        measurements = [qml.state()]
        circuit = QuantumScript(ops, measurements)
        
        # Should not raise an exception
        executor._validate_circuit(circuit)


class TestMeasurementSupportValidation:
    """Test measurement type support validation."""
    
    @pytest.mark.parametrize("backend_type,measurement_type,should_support", [
        (BackendType.LOCAL_SIMULATOR, qml.measurements.ProbabilityMP, True),
        (BackendType.LOCAL_SIMULATOR, qml.measurements.StateMP, True),
        (BackendType.TIANYAN_SIMULATOR, qml.measurements.ProbabilityMP, True),
        (BackendType.TIANYAN_SIMULATOR, qml.measurements.StateMP, False),
        (BackendType.TIANYAN_HARDWARE, qml.measurements.SampleMP, True),
        (BackendType.TIANYAN_HARDWARE, qml.measurements.StateMP, False),
    ])
    def test_measurement_support(self, backend_type, measurement_type, should_support):
        """Test measurement support validation for different backends."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        executor._backend_type = backend_type
        
        measurement = measurement_type()
        
        if should_support:
            # Should not raise exception
            executor._validate_measurement_support(measurement)
        else:
            with pytest.raises(ValueError, match="does not support"):
                executor._validate_measurement_support(measurement)


class TestCircuitExecution:
    """Test circuit execution functionality."""
    
    def test_single_measurement_execution(self):
        """Test execution with single measurement."""
        config = {
            'machine_name': 'default',
            'shots': None,
            'wires': 2
        }
        
        executor = CircuitExecutor(config)
        
        # Mock the internal execution methods
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
            
            # Verify single result is returned (not list)
            assert isinstance(result, np.ndarray)
            assert mock_convert.called
            assert mock_execute.called
            assert mock_measure.called
            
    def test_multiple_measurements_execution(self):
        """Test execution with multiple measurements."""
        config = {
            'machine_name': 'default', 
            'shots': None,
            'wires': 2
        }
        
        executor = CircuitExecutor(config)
        
        with patch.object(executor, '_convert_to_cqlib_format') as mock_convert, \
             patch.object(executor, '_execute_on_backend') as mock_execute, \
             patch.object(executor, '_execute_measurement') as mock_measure:
            
            mock_convert.return_value = (Mock(), "test_qcis")
            mock_execute.return_value = {'probabilities': {'00': 1.0}}
            mock_measure.side_effect = [
                np.array([1.0, 0.0, 0.0, 0.0]),  # First measurement
                1.0  # Second measurement
            ]
            
            ops = [qml.Identity(0)]
            measurements = [qml.probs(), qml.expval(qml.PauliZ(0))]
            circuit = QuantumScript(ops, measurements)
            
            results = executor.execute_circuit(circuit)
            
            # Verify list of results is returned
            assert isinstance(results, list)
            assert len(results) == 2
            assert mock_measure.call_count == 2


class TestMeasurementProcessing:
    """Test measurement-specific result processing."""
    
    def test_probability_measurement_processing(self):
        """Test probability measurement result processing."""
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
        """Test expectation value measurement processing."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        
        raw_result = {
            'probabilities': {'00': 0.5, '11': 0.5}
        }
        
        # Expectation of Z⊗Z on Bell state should be 1.0
        measurement = qml.measurements.ExpectationMP(qml.PauliZ(0) @ qml.PauliZ(1))
        result = executor._execute_measurement_impl(measurement, raw_result)
        
        assert result == 1.0  # <ZZ> = (+1)*0.5 + (+1)*0.5 = 1.0
        
    def test_sample_measurement_processing(self):
        """Test sample measurement result processing."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        
        raw_result = {
            'samples': np.array([[0, 0], [1, 1], [0, 1]])
        }
        
        measurement = qml.measurements.SampleMP()
        result = executor._execute_measurement_impl(measurement, raw_result)
        
        expected = np.array([[0, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(result, expected)
        
    def test_state_measurement_processing(self):
        """Test statevector measurement processing."""
        config = {'machine_name': 'default', 'wires': 1}
        executor = CircuitExecutor(config)
        
        raw_result = {
            'statevector': {'0': 0.70710678, '1': 0.70710678}
        }
        
        measurement = qml.measurements.StateMP()
        result = executor._execute_measurement_impl(measurement, raw_result)
        
        expected = {'0': 0.70710678, '1': 0.70710678}
        assert result == expected


class TestBackendExecution:
    """Test backend-specific execution methods."""
    
    def test_local_simulator_execution(self):
        """Test local simulator execution."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        
        mock_circuit = Mock()
        mock_simulator = Mock()
        
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
            
    def test_tianyan_simulator_execution(self):
        """Test Tianyan simulator execution."""
        config = {
            'machine_name': 'tianyan_s',
            'login_key': 'test_key',
            'shots': 1000,
            'wires': 2
        }
        
        with patch('cqlib_adapter.pennylane_ext.circuit_executor.TianYanPlatform') as mock_platform_class:
            mock_platform = Mock()
            mock_platform_class.return_value = mock_platform
            
            executor = CircuitExecutor(config)
            
            mock_platform.submit_experiment.return_value = 'test_query_id'
            mock_platform.query_experiment.return_value = [{
                'resultStatus': 'S[0.5,0.5]',
                'probability': '{"00":0.5,"11":0.5}'
            }]
            
            result = executor._execute_tianyan_simulator(Mock(), "test_qcis")
            
            assert 'probabilities' in result
            assert 'samples' in result
            mock_platform.submit_experiment.assert_called_once_with(
                "test_qcis", num_shots=1000
            )


class TestResultReordering:
    """Test result reordering functionality."""
    
    def test_probability_reordering(self):
        """Test probability dictionary reordering."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        
        probabilities = {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25}
        wire_labels = [1, 0]  # Swap qubit order
        
        reordered = executor._reorder_probability_dict(probabilities, wire_labels)
        
        # With wire_labels [1,0], bitstring '01' becomes '10' etc.
        assert reordered['00'] == 0.25  # 00 -> 00
        assert reordered['01'] == 0.25  # 01 -> 10
        assert reordered['10'] == 0.25  # 10 -> 01  
        assert reordered['11'] == 0.25  # 11 -> 11
        
    def test_sample_reordering(self):
        """Test sample matrix reordering."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        
        samples = np.array([[0, 1], [1, 0]])  # [[q0,q1], ...]
        wire_labels = [1, 0]  # Swap order
        
        reordered = executor._reorder_sample_matrix(samples, wire_labels)
        
        expected = np.array([[1, 0], [0, 1]])  # [[q1,q0], ...]
        np.testing.assert_array_equal(reordered, expected)


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.parametrize("decimal,bits,little_endian,expected", [
        (5, 4, True, [1, 0, 1, 0]),  # 5 = 1010 (little-endian)
        (5, 4, False, [0, 1, 0, 1]), # 5 = 0101 (big-endian)
        (0, 3, True, [0, 0, 0]),
        (7, 3, True, [1, 1, 1]),
    ])
    def test_decimal_to_binary_array(self, decimal, bits, little_endian, expected):
        """Test decimal to binary array conversion."""
        result = decimal_to_binary_array(decimal, bits, little_endian)
        expected_array = np.array(expected)
        np.testing.assert_array_equal(result, expected_array)
        
    def test_samples_to_pennylane_format(self):
        """Test sample format conversion."""
        samples = [1, 2, 3]  # Decimal samples
        num_qubits = 2
        
        result = samples_to_pennylane_format(samples, num_qubits)
        
        expected = np.array([
            [1, 0],  # 1 = 01 -> [1,0] little-endian
            [0, 1],  # 2 = 10 -> [0,1] little-endian  
            [1, 1],  # 3 = 11 -> [1,1] little-endian
        ])
        np.testing.assert_array_equal(result, expected)
        
    def test_switch_endianness(self):
        """Test endianness switching."""
        data_1d = [1, 0, 1, 0]
        result_1d = switch_endianness(data_1d)
        expected_1d = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(result_1d, expected_1d)
        
        data_2d = [[1, 0], [0, 1]]
        result_2d = switch_endianness(data_2d)
        expected_2d = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(result_2d, expected_2d)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_backend_connection_failure(self):
        """Test handling of backend connection failures."""
        config = {
            'machine_name': 'tianyan_s',
            'login_key': 'test_key',
            'wires': 2
        }
        
        with patch('cqlib_adapter.pennylane_ext.circuit_executor.TianYanPlatform') as mock_platform:
            mock_platform.side_effect = Exception("Connection failed")
            
            with pytest.raises(ConnectionError, match="Backend connection failed"):
                CircuitExecutor(config)
                
    def test_missing_probabilities_in_raw_result(self):
        """Test handling of missing probabilities in raw results."""
        config = {'machine_name': 'default', 'wires': 2}
        executor = CircuitExecutor(config)
        
        raw_result = {}  # No probabilities key
        
        measurement = qml.measurements.ExpectationMP(qml.PauliZ(0))
        
        with pytest.raises(ValueError, match="must contain 'probabilities' key"):
            executor._execute_measurement_impl(measurement, raw_result)


class TestExecutionStatistics:
    """Test execution statistics functionality."""
    
    def test_get_execution_stats(self):
        """Test retrieval of execution statistics."""
        config = {
            'machine_name': 'default',
            'shots': 1000,
            'wires': 5,
            'verbose': True
        }
        
        executor = CircuitExecutor(config)
        
        # Execute some circuits to increment count
        with patch.object(executor, '_convert_to_cqlib_format') as mock_convert, \
             patch.object(executor, '_execute_on_backend') as mock_execute, \
             patch.object(executor, '_execute_measurement') as mock_measure:
            
            # FIX: Provide proper return values for the mock
            mock_circuit_obj = Mock()
            mock_convert.return_value = (mock_circuit_obj, "test_qcis")
            mock_execute.return_value = {'probabilities': {'0': 1.0}}
            mock_measure.return_value = np.array([1.0, 0.0])
            
            ops = [qml.Hadamard(0)]
            measurements = [qml.probs()]
            circuit = QuantumScript(ops, measurements)
            
            executor.execute_circuit(circuit)
            executor.execute_circuit(circuit)
            
        stats = executor.get_execution_stats()
        
        assert stats['execution_count'] == 2
        assert stats['backend_type'] == 'local'
        assert stats['wires'] == 5
        assert stats['shots'] == 1000
        assert stats['machine_name'] == 'default'


# Integration test for complete workflow
class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_local_execution_workflow(self):
        """Test complete execution workflow with local simulator."""
        config = {
            'machine_name': 'default',
            'shots': None,
            'wires': 2,
            'verbose': False
        }
        
        executor = CircuitExecutor(config)
        
        # Create a simple circuit
        ops = [
            qml.Hadamard(0),
            qml.CNOT([0, 1])
        ]
        measurements = [qml.probs()]
        circuit = QuantumScript(ops, measurements)
        
        # Mock the CQLib components

        with patch('cqlib.utils.qasm2') as mock_qasm2, \
             patch('cqlib.simulator.statevector_simulator.StatevectorSimulator') as mock_simulator_class:
            
            mock_circuit = Mock()
            mock_circuit.qcis = "test_qcis"
            mock_qasm2.loads.return_value = mock_circuit
            
            mock_simulator = Mock()
            mock_simulator_class.return_value = mock_simulator
            mock_simulator.probs.return_value = {'00': 0.5, '11': 0.5}
            mock_simulator.sample.return_value = np.array([0, 3])  # Decimal samples
            mock_simulator.statevector.return_value = {'00': 0.707, '11': 0.707}
            
            result = executor.execute_circuit(circuit)
            
            # Verify the result is a probability array
            assert isinstance(result, np.ndarray)
            assert len(result) == 4  # 2^2 = 4 probabilities
            
    def test_logging_setup(self):
        """Test that logging is properly configured."""
        config = {
            'machine_name': 'default',
            'shots': 1000,
            'wires': 2,
            'verbose': True
        }
        
        executor = CircuitExecutor(config)
        
        # Check that logger is configured
        assert executor.logger is not None
        assert executor.logger.level == logging.INFO
        
        # Test with verbose disabled
        config['verbose'] = False
        executor = CircuitExecutor(config)
        # Logger should exist but may not have handlers
        assert executor.logger is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])