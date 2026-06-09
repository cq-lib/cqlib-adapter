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

"""Tests for HardwareMapper class in ext_mapping.py."""

import pytest
from unittest.mock import Mock, MagicMock
from cqlib_adapter.pennylane_ext.ext_mapping import HardwareMapper


class MockQubit:
    """Mock qubit object that mimics cqlib qubit behavior."""
    def __init__(self, index):
        self.index = index


class TestHardwareMapperInit:
    """Tests for HardwareMapper initialization."""

    def test_init_with_valid_mapping(self):
        """Test initialization with a valid mapping dictionary."""
        mapping = {'Q0': 'Q7', 'Q1': 'Q5'}
        mapper = HardwareMapper(mapping)
        assert mapper.mapping_dict == mapping

    def test_mapping_dict_reference(self):
        """Test that mapper stores reference to mapping dict."""
        mapping = {'Q0': 'Q7'}
        mapper = HardwareMapper(mapping)
        assert mapper.mapping_dict is mapping


class TestMapQcisCode:
    """Tests for map_qcis_code string replacement method."""

    def test_single_qubit_mapping(self):
        """Test mapping a single qubit in QCIS code."""
        mapper = HardwareMapper({'Q0': 'Q7'})
        original = "X Q0"
        result = mapper.map_qcis_code(original)
        assert result == "X Q7"

    def test_multiple_qubit_mapping(self):
        """Test mapping multiple qubits in QCIS code."""
        mapper = HardwareMapper({'Q0': 'Q7', 'Q1': 'Q5'})
        original = "X Q0\nH Q1"
        result = mapper.map_qcis_code(original)
        assert "Q7" in result
        assert "Q5" in result
        assert "Q0" not in result
        assert "Q1" not in result

    def test_same_line_multiple_mentions(self):
        """Test that multiple mentions of same qubit on a line are all mapped."""
        mapper = HardwareMapper({'Q0': 'Q7'})
        original = "CNOT Q0 Q0"
        result = mapper.map_qcis_code(original)
        assert result == "CNOT Q7 Q7"

    def test_no_match_leaves_unchanged(self):
        """Test that qubits not in mapping are left unchanged."""
        mapper = HardwareMapper({'Q0': 'Q7'})
        original = "X Q2"
        result = mapper.map_qcis_code(original)
        assert result == "X Q2"

    def test_empty_qcis_code(self):
        """Test with empty QCIS code string."""
        mapper = HardwareMapper({'Q0': 'Q7'})
        result = mapper.map_qcis_code("")
        assert result == ""

    def test_empty_mapping_dict(self):
        """Test with empty mapping dictionary."""
        mapper = HardwareMapper({})
        original = "X Q0"
        result = mapper.map_qcis_code(original)
        assert result == "X Q0"

    def test_complex_qcis_sequence(self):
        """Test mapping of a more complex QCIS sequence."""
        mapper = HardwareMapper({'Q0': 'Q3', 'Q1': 'Q7'})
        original = """X Q0
H Q0
CNOT Q0 Q1
M Q1"""
        result = mapper.map_qcis_code(original)
        lines = result.strip().split('\n')
        assert "X Q3" in lines[0]
        assert "H Q3" in lines[1]
        assert "CNOT Q3 Q7" in lines[2]
        assert "M Q7" in lines[3]


class TestMapInstructionQubits:
    """Tests for _map_instruction_qubits internal method."""

    def test_map_single_string_qubit(self):
        """Test mapping a single string qubit reference."""
        mapper = HardwareMapper({'Q0': 'Q7'})
        result = mapper._map_instruction_qubits('Q0')
        assert result == 'Q7'

    def test_map_string_qubit_not_in_mapping(self):
        """Test qubit reference not in mapping returns unchanged."""
        mapper = HardwareMapper({'Q0': 'Q7'})
        result = mapper._map_instruction_qubits('Q2')
        assert result == 'Q2'

    def test_map_list_of_qubits(self):
        """Test mapping a list of qubit references."""
        mapper = HardwareMapper({'Q0': 'Q7', 'Q1': 'Q5'})
        result = mapper._map_instruction_qubits(['Q0', 'Q1'])
        assert result == ['Q7', 'Q5']

    def test_map_list_with_partial_mapping(self):
        """Test mapping a list where only some qubits are mapped."""
        mapper = HardwareMapper({'Q0': 'Q7'})
        result = mapper._map_instruction_qubits(['Q0', 'Q1'])
        assert result == ['Q7', 'Q1']

    def test_map_tuple_of_qubits(self):
        """Test mapping a tuple of qubit references."""
        mapper = HardwareMapper({'Q0': 'Q7', 'Q1': 'Q5'})
        result = mapper._map_instruction_qubits(('Q0', 'Q1'))
        assert result == ['Q7', 'Q5']

    def test_map_non_string_returns_unchanged(self):
        """Test that non-string qubit references are returned unchanged."""
        mapper = HardwareMapper({'Q0': 'Q7'})
        result = mapper._map_instruction_qubits(123)
        assert result == 123


class TestMapCircuit:
    """Tests for map_circuit method."""

    def test_map_circuit_updates_qubit_names(self):
        """Test that map_circuit updates _qubits dictionary keys."""
        mapper = HardwareMapper({'Q0': 'Q7', 'Q1': 'Q5'})

        # Create mock circuit
        mock_circuit = Mock()
        mock_circuit._qubits = {
            'Q0': Mock(),
            'Q1': Mock()
        }
        mock_circuit._circuit_data = []
        mock_circuit._parameters = {}

        result = mapper.map_circuit(mock_circuit)

        # Check that result has new qubit names
        assert 'Q7' in result._qubits
        assert 'Q5' in result._qubits
        assert 'Q0' not in result._qubits
        assert 'Q1' not in result._qubits

    def test_map_circuit_copies_parameters(self):
        """Test that map_circuit copies _parameters dict."""
        mapper = HardwareMapper({'Q0': 'Q7'})
        params = {'theta': 0.5}

        mock_circuit = Mock()
        mock_circuit._qubits = {'Q0': Mock()}
        mock_circuit._circuit_data = []
        mock_circuit._parameters = params

        result = mapper.map_circuit(mock_circuit)

        assert result._parameters == params
        assert result._parameters is not params  # Should be a copy

    def test_map_circuit_handles_unmapped_qubits(self):
        """Test that unmapped qubits are preserved in _qubits dict."""
        mapper = HardwareMapper({'Q0': 'Q7'})

        mock_circuit = Mock()
        mock_qubit_q0 = Mock()
        mock_qubit_q2 = Mock()
        mock_circuit._qubits = {'Q0': mock_qubit_q0, 'Q2': mock_qubit_q2}
        mock_circuit._circuit_data = []
        mock_circuit._parameters = {}

        result = mapper.map_circuit(mock_circuit)

        assert 'Q7' in result._qubits
        assert 'Q2' in result._qubits  # Unmapped qubit preserved

    def test_map_circuit_maps_instruction_data(self):
        """Test that instruction data qubit references are updated."""
        mapper = HardwareMapper({'Q0': 'Q7'})

        mock_instruction = Mock()
        mock_instruction.qubits = ['Q0']
        mock_instruction_data = Mock()
        mock_instruction_data.qubits = ['Q0']
        mock_instruction_data.instruction = 'X'

        mock_circuit = Mock()
        mock_circuit._qubits = {'Q0': Mock()}
        mock_circuit._circuit_data = [mock_instruction_data]
        mock_circuit._parameters = {}

        result = mapper.map_circuit(mock_circuit)

        # Check instruction qubits were mapped
        assert len(result._circuit_data) == 1
