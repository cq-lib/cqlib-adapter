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


import sys
import numpy as np
from typing import Union
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterExpression
from cqlib import Circuit
from cqlib import Parameter as CqlibParameter
import pennylane as qml
from pennylane.tape import QuantumTape

def qiskit_to_cqlib(
    qiskit_qc: QuantumCircuit, 
    mapping: Union[dict, list] = None, 
    decompose: bool = False
) -> Circuit:
    """
    Convert a Qiskit QuantumCircuit into a cqlib Circuit.
    
    Features:
    - Native support for complex parameter expressions.
    - Recursive custom gate unwrapping.
    - Buffered mid-circuit measurement handling to strictly preserve endianness.
    - Strict Virtual-to-Physical Qubit Mapping (Layout routing) via physical qubit lists.
    
    Args:
        qiskit_qc (QuantumCircuit): The source Qiskit quantum circuit.
        mapping (dict or list, optional): Mapping from Qiskit virtual qubit indices to 
                                          cqlib physical qubit indices. Must contain all 
                                          qubits used in the circuit if provided.
                                          E.g., {0: 7, 1: 3, 2: 1} or [7, 3, 1]. Defaults to None.
        decompose (bool): If True, transpiles the circuit to hardware basis gates.
        
    Returns:
        Circuit: The converted cqlib Circuit instance.
        
    Raises:
        ValueError: If a qubit index is missing from the provided mapping.
    """

    if decompose:
        basis_gates = ['x', 'y', 'z','s','sdg','t', 'tdg', 'sx', 'sxdg', 'id', 'rx', 'ry', 'rz', 'p', 'cz', 'barrier', 'delay','measure']
        qiskit_qc = transpile(qiskit_qc, basis_gates=basis_gates)
        
    # ==========================================
    # 2. Parameter Initialization
    # ==========================================
    param_mapping = {}  
    cqlib_params = []   
    
    for q_param in qiskit_qc.parameters:
        cq_param = CqlibParameter(q_param.name)
        param_mapping[q_param.name] = cq_param
        cqlib_params.append(cq_param)

    # ==========================================
    # 3. Strict Virtual-to-Physical Qubit Mapping
    # ==========================================
    top_level_qubit_map = {}
    physical_qubits_list = []  # To store the exact layout, e.g., [7, 3, 1]
    
    for q in qiskit_qc.qubits:
        v_idx = qiskit_qc.find_bit(q).index
        
        # Determine the target physical index based on user mapping
        if mapping is not None:
            if isinstance(mapping, dict):
                # Strict check: raise error if v_idx is missing from the dictionary
                if v_idx not in mapping:
                    raise ValueError(
                        f"Virtual qubit index {v_idx} is missing from the provided mapping dictionary. "
                        f"Please ensure all qubits in the circuit are mapped."
                    )
                p_idx = mapping[v_idx]
            elif isinstance(mapping, (list, tuple)):
                # Strict check: raise error if the list doesn't cover all qubits
                if v_idx >= len(mapping):
                    raise ValueError(
                        f"Virtual qubit index {v_idx} exceeds the length of the provided mapping list. "
                        f"Please provide a mapping for every qubit."
                    )
                p_idx = mapping[v_idx]
            else:
                raise TypeError("The 'mapping' parameter must be a dict or a list/tuple.")
        else:
            p_idx = v_idx
            
        top_level_qubit_map[q] = p_idx
        physical_qubits_list.append(p_idx)


    cqlib_qc = Circuit(physical_qubits_list, parameters=cqlib_params)
    
    # --- Helper: Parse Mathematical Parameters ---
    def parse_param(q_param):
        if isinstance(q_param, ParameterExpression):
            if len(q_param.parameters) == 0:
                return float(q_param)
            elif len(q_param.parameters) == 1 and q_param == list(q_param.parameters)[0]:
                return param_mapping[list(q_param.parameters)[0].name]
            else:
                expr_str = str(q_param)
                local_env = {name: param for name, param in param_mapping.items()}
                try:
                    return eval(expr_str, {"__builtins__": None}, local_env)
                except Exception as e:
                    raise ValueError(f"Failed to translate math expression '{expr_str}'. Error: {e}")
        else:
            return float(q_param)

    # ==========================================
    # 4. Recursive Instruction Processor
    # ==========================================
    def process_circuit(circuit_to_parse: QuantumCircuit, qubit_map: dict, clbit_map: dict = None):
        """Recursively unroll instructions, custom gates, and buffered measurements."""
        measure_buffer = {}

        def flush_measurements():
            if measure_buffer:
                sorted_c_indices = sorted(measure_buffer.keys())
                ordered_q_indices = [measure_buffer[c_idx] for c_idx in sorted_c_indices]
                cqlib_qc.measure(ordered_q_indices)
                measure_buffer.clear()

        for instruction in circuit_to_parse.data:
            op = instruction.operation
            name = op.name.lower()
            
            # --- Measurement Handling ---
            if name == 'measure':
                q_obj = instruction.qubits[0]
                c_obj = instruction.clbits[0]
                q_idx = qubit_map[q_obj]
                
                if clbit_map is not None and c_obj in clbit_map:
                    c_idx = clbit_map[c_obj]
                else:
                    c_idx = qiskit_qc.find_bit(c_obj).index
                    
                measure_buffer[c_idx] = q_idx
                continue 
            
            # Non-Measurement Gate encountered: Flush pending measurements first
            flush_measurements()
            
            # Map inner virtual qubits to absolute physical indices using the current dictionary
            q_indices = [qubit_map[q] for q in instruction.qubits]
            
            if name == 'h': cqlib_qc.h(q_indices[0])
            elif name == 'x': cqlib_qc.x(q_indices[0])
            elif name == 'y': cqlib_qc.y(q_indices[0])
            elif name == 'z': cqlib_qc.z(q_indices[0])
            elif name == 's': cqlib_qc.s(q_indices[0])
            elif name == 'sdg': cqlib_qc.sd(q_indices[0])
            elif name == 't': cqlib_qc.t(q_indices[0])
            elif name == 'tdg': cqlib_qc.td(q_indices[0])
            elif name == 'sx': cqlib_qc.x2p(q_indices[0])
            elif name == 'sxdg': cqlib_qc.x2m(q_indices[0])
            elif name == 'id':
                pass 


            elif name == 'delay':
                raw_duration = op.duration
                
                if op.unit:
                    unit = op.unit.lower()
                    parsed_duration = parse_param(raw_duration)
                    
                    if unit == 'dt':
                        raise ValueError(
                            "Unsupported delay unit 'dt'. Unit 'dt' requires explicit backend sampling rate. "
                            "Please use 'ns', 'us', 'ms', or 's'."
                        )
                    elif unit == 'ps':
                        raise ValueError(
                            "Unsupported delay unit 'ps'. Unit 'ps' exceeds hardware timing resolution. "
                            "Please use 'ns', 'us', 'ms', or 's'."
                        )
                    elif unit == 'ns':
                        cqlib_duration = parsed_duration
                    elif unit == 'us':
                        cqlib_duration = parsed_duration * 1e3
                    elif unit == 'ms':
                        cqlib_duration = parsed_duration * 1e6
                    elif unit == 's':
                        cqlib_duration = parsed_duration * 1e9
                    else:
                        raise ValueError(f"Unknown delay unit: '{unit}'.")
                
                else:
                    raise ValueError(
                        "Default unit of delay is 'dt', which requires explicit backend sampling rate. "
                        "Please explicitly input one of the following units: 'ns', 'us' (for mu s), 'ms', 's'."
                    )

                for q_idx in q_indices:
                    cqlib_qc.i(q_idx, cqlib_duration)

                
            elif name == 'rx': cqlib_qc.rx(q_indices[0], theta=parse_param(op.params[0]))
            elif name == 'ry': cqlib_qc.ry(q_indices[0], theta=parse_param(op.params[0]))
            elif name in ['rz', 'p']: cqlib_qc.rz(q_indices[0], theta=parse_param(op.params[0]))
            elif name in ['u3', 'u']:
                cqlib_qc.u(q_indices[0], 
                           theta=parse_param(op.params[0]), 
                           phi=parse_param(op.params[1]), 
                           lam=parse_param(op.params[2]))
                
            elif name == 'cx': cqlib_qc.cx(q_indices[0], q_indices[1])
            elif name == 'cy': cqlib_qc.cy(q_indices[0], q_indices[1])
            elif name == 'cz': cqlib_qc.cz(q_indices[0], q_indices[1])
            elif name == 'ccx': cqlib_qc.ccx(q_indices[0], q_indices[1], q_indices[2])
            elif name == 'swap': cqlib_qc.swap(q_indices[0], q_indices[1])
            
            elif name == 'crx': cqlib_qc.crx(q_indices[0], q_indices[1], theta=parse_param(op.params[0]))
            elif name == 'cry': cqlib_qc.cry(q_indices[0], q_indices[1], theta=parse_param(op.params[0]))
            elif name == 'crz': cqlib_qc.crz(q_indices[0], q_indices[1], theta=parse_param(op.params[0]))
                
            elif name == 'barrier': 
                cqlib_qc.barrier(*q_indices)
            
            # Custom Gate Unwrapping
            elif hasattr(op, 'definition') and op.definition is not None:
                inner_qc = op.definition
                # Propagate physical mappings down to the custom gate's internal virtual qubits
                inner_qubit_map = {inner_q: q_indices[i] for i, inner_q in enumerate(inner_qc.qubits)}
                process_circuit(inner_qc, inner_qubit_map, clbit_map) 
            else:
                print(f"Warning: Gate '{name}' is unsupported and has no definition. Skipped.")

        # Final flush for any trailing measurements
        flush_measurements()

    # ==========================================
    # 5. Initiate Parsing
    # ==========================================
    top_level_clbit_map = {c: qiskit_qc.find_bit(c).index for c in qiskit_qc.clbits}
    
    # Process using the dynamically routed top_level_qubit_map
    process_circuit(qiskit_qc, top_level_qubit_map, top_level_clbit_map)
            
    return cqlib_qc


def pennylane_to_cqlib(
    tape: QuantumTape, 
    mapping: Union[dict, list] = None,
    parameters: list = None,
    decompose: bool = False
) -> Circuit:
    """
    Convert a PennyLane QuantumTape into a cqlib Circuit.
    
    Features:
    - Virtual-to-Physical Qubit Mapping (Layout routing).
    - Explicit parameter injection (zero-overhead) or Auto-extraction.
    - Smart Hardware Basis Interceptor (decompose=True) routes topology-incompatible 
      gates (CNOT, CY) and basis-incompatible gates (H) to the hardware equivalents,
      while preserving the whitelisted native set (X, Y, Z, S, T, SX, RZ, CZ, ID).
    - Recursive unwrapping of PennyLane QML Templates and complex gates.
    - Robust Mid-Circuit Measurement (MidMeasureMP) support.
    
    Args:
        tape (QuantumTape): The source PennyLane quantum tape.
        mapping (dict or list, optional): Mapping from virtual wires to physical qubits.
        parameters (list, optional): Explicit list of cqlib.Parameter instances.
        decompose (bool): If True, strictly transpiles the circuit to the native hardware basis.
    """
    
    # ==========================================
    # 1. Parameter Handling (Explicit vs Implicit)
    # ==========================================
    cqlib_params = []
    
    if parameters is not None:
        cqlib_params = parameters
    else:
        def collect_params(p_obj):
            if isinstance(p_obj, CqlibParameter):
                if p_obj not in cqlib_params:
                    cqlib_params.append(p_obj)
            elif isinstance(p_obj, (list, tuple, np.ndarray)):
                for item in np.array(p_obj).flatten():
                    collect_params(item)
            elif hasattr(p_obj, 'numpy'):
                try:
                    for item in np.array(p_obj.numpy()).flatten():
                        collect_params(item)
                except Exception:
                    pass

       
        for op in tape.operations:
            if hasattr(op, 'parameters'):
                for p in op.parameters:
                    collect_params(p)
            
    # ==========================================
    # 2. Virtual-to-Physical Wire Mapping
    # ==========================================
    original_wires = tape.wires.tolist()
    top_level_qubit_map = {}
    physical_qubits_list = []
    
    for v_idx, wire in enumerate(original_wires):
        if mapping is not None:
            if isinstance(mapping, dict):
                if wire in mapping:
                    p_idx = mapping[wire]
                elif v_idx in mapping:
                    p_idx = mapping[v_idx]
                else:
                    raise ValueError(f"Wire '{wire}' is missing from the mapping dictionary.")
            elif isinstance(mapping, (list, tuple)):
                if v_idx >= len(mapping):
                    raise ValueError(f"Wire index {v_idx} exceeds mapping list length.")
                p_idx = mapping[v_idx]
        else:
            p_idx = wire if isinstance(wire, int) else v_idx
            
        top_level_qubit_map[wire] = p_idx
        if p_idx not in physical_qubits_list:
            physical_qubits_list.append(p_idx)

    cqlib_qc = Circuit(physical_qubits_list, parameters=cqlib_params)
    
    def parse_param(p):
        if isinstance(p, CqlibParameter):
            return p
        if hasattr(p, 'numpy'):
            p = p.numpy()
        try:
            return float(p)
        except (TypeError, ValueError):
            return p

    # ==========================================
    # 3. Recursive Operation Processor & Decomposer
    # ==========================================
    measure_buffer = []

    def flush_measurements():
        if measure_buffer:
            cqlib_qc.measure(measure_buffer)
            measure_buffer.clear()

    def process_operations(operations):
        for op in operations:
            name = op.name
            wires = op.wires.tolist()
            
    
            raw_params = getattr(op, 'parameters', [])
            params = [parse_param(p) for p in raw_params]
            
            # --- Measurement Handling ---
            if name == 'MidMeasureMP':
                q_indices = [top_level_qubit_map[w] for w in wires]
                measure_buffer.extend(q_indices)
                continue
            
            flush_measurements()
            

            if decompose:
                if name == 'CNOT':
                    sub_ops = [
                        qml.Hadamard(wires=wires[1]),
                        qml.CZ(wires=[wires[0], wires[1]]),
                        qml.Hadamard(wires=wires[1])
                    ]
                    process_operations(sub_ops)
                    continue
                elif name == 'CY':
                    sub_ops = [
                        qml.adjoint(qml.S(wires=wires[1])),
                        qml.CNOT(wires=[wires[0], wires[1]]), 
                        qml.S(wires=wires[1])
                    ]
                    process_operations(sub_ops)
                    continue
                elif name == 'Hadamard':
                    sub_ops = [
                        qml.RZ(np.pi/2, wires=wires[0]),
                        qml.SX(wires=wires[0]),
                        qml.RZ(np.pi/2, wires=wires[0])
                    ]
                    process_operations(sub_ops)
                    continue
                
                hardware_basis = {
                    'PauliX', 'PauliY', 'PauliZ', 'RX', 'RY','RZ', 'CZ', 'Identity', 'SX', 'Adjoint(SX)', 'Barrier', 'PhaseShift',
                    'PauliZ', 'S', 'Adjoint(S)', 'T', 'Adjoint(T)'
                }
                
                if name not in hardware_basis and op.has_decomposition:
                    process_operations(op.decomposition())
                    continue

            # ==========================================
            # --- Standard Gates Mapping ---
            # ==========================================
            q_indices = [top_level_qubit_map[w] for w in wires]
            
            if name == 'PauliZ': cqlib_qc.z(q_indices[0])
            elif name == 'S': cqlib_qc.s(q_indices[0])
            elif name == 'Adjoint(S)': cqlib_qc.sd(q_indices[0])
            elif name == 'T': cqlib_qc.t(q_indices[0])
            elif name == 'Adjoint(T)': cqlib_qc.td(q_indices[0])
            
            elif name == 'Hadamard': cqlib_qc.h(q_indices[0])
            elif name == 'PauliX': cqlib_qc.x(q_indices[0])
            elif name == 'PauliY': cqlib_qc.y(q_indices[0])
            elif name == 'SX': cqlib_qc.x2p(q_indices[0])
            elif name == 'Adjoint(SX)': cqlib_qc.x2m(q_indices[0])
            elif name == 'Identity': pass
                
            elif name == 'RX': cqlib_qc.rx(q_indices[0], theta=params[0])
            elif name == 'RY': cqlib_qc.ry(q_indices[0], theta=params[0])
            elif name in ['RZ', 'PhaseShift']: cqlib_qc.rz(q_indices[0], theta=params[0])
            
            elif name == 'CNOT': cqlib_qc.cx(q_indices[0], q_indices[1])
            elif name == 'CY': cqlib_qc.cy(q_indices[0], q_indices[1])
            elif name == 'CZ': cqlib_qc.cz(q_indices[0], q_indices[1])
            elif name == 'Toffoli': cqlib_qc.ccx(q_indices[0], q_indices[1], q_indices[2])
            elif name == 'SWAP': cqlib_qc.swap(q_indices[0], q_indices[1])
            
            elif name == 'CRX': cqlib_qc.crx(q_indices[0], q_indices[1], theta=params[0])
            elif name == 'CRY': cqlib_qc.cry(q_indices[0], q_indices[1], theta=params[0])
            elif name == 'CRZ': cqlib_qc.crz(q_indices[0], q_indices[1], theta=params[0])
                
            elif name == 'Barrier': 
                cqlib_qc.barrier(*q_indices)
            
            elif op.has_decomposition:
                process_operations(op.decomposition())
            else:
                print(f"Warning: Gate '{name}' is unsupported and has no decomposition. Skipped.")

    # ==========================================
    # 4. Initiate Parsing & Terminal Measurements
    # ==========================================
    process_operations(tape.operations)
    flush_measurements()
    
    final_measured_wires = []
    for m in tape.measurements:
        if m.wires:
            final_measured_wires.extend([top_level_qubit_map[w] for w in m.wires.tolist()])
        else:
            final_measured_wires.extend([top_level_qubit_map[w] for w in original_wires])

    unique_final_meas = []
    for q in final_measured_wires:
        if q not in unique_final_meas:
            unique_final_meas.append(q)

    if unique_final_meas:
        cqlib_qc.measure(unique_final_meas)
        
    return cqlib_qc