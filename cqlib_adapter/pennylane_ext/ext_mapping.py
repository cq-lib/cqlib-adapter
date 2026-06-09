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


from cqlib.circuits.instruction_data import InstructionData


class HardwareMapper:
    """A mapper for adapting quantum circuits to specific hardware configurations.
    
    This class provides functionality to map logical qubits to physical qubits
    in quantum circuits, enabling hardware-specific optimizations and adaptations.
    
    Attributes:
        mapping_dict (dict): A dictionary mapping logical qubit names to physical
            qubit names. Example: {'Q0': 'Q7', 'Q1': 'Q5'}.
    """
    
    def __init__(self, mapping_dict):
        """Initializes the HardwareMapper with a qubit mapping dictionary.
        
        Args:
            mapping_dict (dict): A dictionary mapping logical qubit names to 
                physical qubit names. Example: {'Q0': 'Q7', 'Q1': 'Q5'}.
                
        Raises:
            ValueError: If mapping_dict is empty or None.
        """
        self.mapping_dict = mapping_dict
    
    def map_circuit(self, circuit):
        """Maps a quantum circuit to hardware-specific qubit layout.
        
        Applies the qubit mapping to both the circuit's qubit definitions and
        all instructions in the circuit data.
        
        Args:
            circuit: The quantum circuit object to be mapped. Must have attributes
                _qubits, _circuit_data, and _parameters.
                
        Returns:
            A new circuit object with the same type as the input circuit, but
            with all qubit references updated according to the mapping dictionary.
            
        Raises:
            AttributeError: If the input circuit doesn't have required attributes.
            TypeError: If the circuit type cannot be instantiated properly.
        """
        # Map quantum bit definitions
        mapped_qubits = self._map_qubits(circuit._qubits)
        
        # Map circuit instruction data
        mapped_circuit_data = self._map_instructions_properly(circuit._circuit_data)
        
        # Create new circuit instance with mapped configuration
        num_qubits = len(mapped_qubits)
        mapped_circuit = type(circuit)(num_qubits)
        
        # Apply mapped attributes to new circuit
        mapped_circuit._qubits = mapped_qubits
        mapped_circuit._circuit_data = mapped_circuit_data
        mapped_circuit._parameters = circuit._parameters.copy()
        
        return mapped_circuit
    
    def _map_qubits(self, qubits_dict):
        """Maps a dictionary of qubit objects according to the mapping configuration.
        
        Args:
            qubits_dict (dict): Dictionary mapping qubit names to qubit objects.
                Example: {'Q0': Qubit(0), 'Q1': Qubit(1)}.
                
        Returns:
            dict: A new dictionary with qubit names mapped to physical qubit names,
                preserving the original qubit object types but with updated indices.
        """
        mapped_qubits = {}
        for logical_qubit, qubit_obj in qubits_dict.items():
            if logical_qubit in self.mapping_dict:
                physical_qubit = self.mapping_dict[logical_qubit]
                # Create new qubit object with physical qubit index
                mapped_qubits[physical_qubit] = type(qubit_obj)(int(physical_qubit[1:]))
            else:
                mapped_qubits[logical_qubit] = qubit_obj
        
        return mapped_qubits
    
    def _map_instructions_properly(self, circuit_data):
        """Maps circuit instruction data while preserving InstructionData structure.
        
        Args:
            circuit_data (list): List of InstructionData objects representing
                the quantum circuit's instructions.
                
        Returns:
            list: A new list of InstructionData objects with qubit references
                updated according to the mapping configuration.
                
        Raises:
            AttributeError: If InstructionData objects don't have expected attributes.
        """
        mapped_instructions = []
        
        for instruction_data in circuit_data:
            # Map qubit references in the instruction
            mapped_qubits = self._map_instruction_qubits(instruction_data.qubits)
            
            # Create new InstructionData with mapped qubits
            mapped_instruction = InstructionData(
                instruction=instruction_data.instruction,
                qubits=mapped_qubits
            )
            
            mapped_instructions.append(mapped_instruction)
        
        return mapped_instructions
    
    def _map_instruction_qubits(self, original_qubits):
        """Maps qubit references in individual instructions.
        
        Handles different types of qubit references including single qubits,
        lists of qubits, and tuples of qubits.
        
        Args:
            original_qubits: The original qubit reference(s) from an instruction.
                Can be a string (single qubit), list, or tuple.
                
        Returns:
            The mapped qubit reference(s) with the same structure as the input,
            but with logical qubit names replaced by physical qubit names.
        """
        if isinstance(original_qubits, (list, tuple)):
            # Handle multiple qubits in list or tuple
            mapped_qubits = []
            for qubit_ref in original_qubits:
                if isinstance(qubit_ref, str) and qubit_ref in self.mapping_dict:
                    mapped_qubits.append(self.mapping_dict[qubit_ref])
                else:
                    mapped_qubits.append(qubit_ref)
            return mapped_qubits
        elif isinstance(original_qubits, str):
            # Handle single qubit reference
            if original_qubits in self.mapping_dict:
                return self.mapping_dict[original_qubits]
            else:
                return original_qubits
        else:
            # Return unchanged for other types
            return original_qubits

    def map_qcis_code(self, qcis_code):
        """Directly maps QCIS code string using string replacement.
        
        This method provides a lightweight alternative to full circuit mapping
        when only the QCIS code output is needed.
        
        Args:
            qcis_code (str): The original QCIS code string to be mapped.
            
        Returns:
            str: The mapped QCIS code with logical qubit names replaced by
                physical qubit names.
                
        Example:
            >>> mapper = HardwareMapper({'Q0': 'Q7', 'Q1': 'Q5'})
            >>> original = "X Q1\\nH Q1\\nM Q0"
            >>> mapped = mapper.map_qcis_code(original)
            >>> print(mapped)
            "X Q5\\nH Q5\\nM Q7"
        """
        mapped_qcis = qcis_code
        for logical, physical in self.mapping_dict.items():
            mapped_qcis = mapped_qcis.replace(logical, physical)
        return mapped_qcis