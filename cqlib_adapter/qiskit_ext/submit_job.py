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

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager
from typing import Dict, List
import numpy as np

def run_qasm_on_quantum_computer(
    qasm_string: str,
    token: str = 'your_token',
    backend_name: str = "ibmq_qasm_simulator",
    shots: int = 5000,
    optimization_level: int = 1
) -> Dict:
    """
    Execute QASM code directly on IBM Quantum Platform and return measurement results
    
    Parameters:
    -----------
    qasm_string : str
        Quantum circuit code in OpenQASM format
    token : str, optional
        IBM Quantum API token
    backend_name : str, optional
        Name of the backend to use, defaults to simulator
    shots : int, optional
        Number of measurement shots, defaults to 5000
    optimization_level : int, optional
        Circuit optimization level (0-3), defaults to 1
    
    Returns:
    --------
    Dict
        Dictionary containing computation results:
        - 'success': Whether the execution was successful
        - 'counts': Measurement result counts statistics
        - 'job_id': Job ID
        - 'backend': Backend used
        - 'circuit_qubits': Number of circuit qubits
        - 'shots_used': Actual number of shots used
        - 'message': Status message
    """
    
    try:
        # Step 1: Create quantum circuit from QASM string
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        num_qubits = circuit.num_qubits
        
        print(f"✓ QASM parsing successful: {num_qubits} qubit circuit")
        
        # Step 2: Initialize service
        service = QiskitRuntimeService(token=token)
        backend = service.backend(backend_name)
        print(f"✓ Using backend: {backend.name}")
        
        # Step 3: Optimize circuit
        pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
        isa_circuit = pm.run(circuit)
        
        # Step 4: Create Sampler and submit job
        sampler = Sampler(backend=backend)
        sampler.options.default_shots = shots
        
        job = sampler.run([isa_circuit])
        print(f"✓ Job submitted, ID: {job.job_id()}")
        print("⏳ Waiting for computation to complete...")
        
        # Step 5: Get results
        job_result = job.result()
        pub_result = job_result[0]
        
        # Step 6: Extract measurement results
        # Get all possible bit strings and their probabilities/counts
        if hasattr(pub_result.data, 'meas'):
            # New version API
            counts = pub_result.data.meas.get_counts()
        else:
            # Compatibility with older API versions
            counts = {}
            for bitstring, prob in enumerate(pub_result.data.meas):
                if prob > 0:
                    # Convert probability to count
                    count = int(prob * shots)
                    if count > 0:
                        # Convert index to binary string
                        bit_str = format(bitstring, f'0{num_qubits}b')
                        counts[bit_str] = count
        
        print("✅ Computation completed!")
        
        # Return results
        return {
            'success': True,
            'counts': counts,
            'job_id': job.job_id(),
            'backend': backend_name,
            'circuit_qubits': num_qubits,
            'shots_used': shots,
            'message': 'Quantum computation completed successfully'
        }
        
    except Exception as e:
        error_msg = f'Quantum computation failed: {str(e)}'
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'counts': {},
            'job_id': None,
            'backend': backend_name,
            'circuit_qubits': 0,
            'shots_used': shots,
            'message': error_msg
        }

def print_quantum_results(results: Dict):
    """
    Print formatted quantum computation results
    """
    if not results['success']:
        print(f"Error: {results['message']}")
        return
    
    print("\n" + "="*50)
    print("🔬 Quantum Measurement Results")
    print("="*50)
    print(f"Job ID: {results['job_id']}")
    print(f"Backend: {results['backend']}")
    print(f"Number of Qubits: {results['circuit_qubits']}")
    print(f"Number of Shots: {results['shots_used']}")
    print("\nMeasurement Results Statistics:")
    print("-" * 30)
    
    counts = results['counts']
    total_shots = sum(counts.values())
    
    # Sort by count and display
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for bitstring, count in sorted_counts:
        percentage = (count / total_shots) * 100
        print(f"{bitstring}: {count:4d} shots ({percentage:5.1f}%)")

# Usage example
if __name__ == "__main__":
    print("=== Quantum Computing Service Demo (No Measurement Operators) ===\n")
    
    # Example 1: Bell state circuit
    bell_qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """
    
    print("1. Running Bell state circuit:")
    result1 = run_qasm_on_quantum_computer(bell_qasm)
    print_quantum_results(result1)
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Superposition state circuit
    superposition_qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    h q[0];
    measure q[0] -> c[0];
    """
    
    print("2. Running single-qubit superposition:")
    result2 = run_qasm_on_quantum_computer(superposition_qasm, shots=2000)
    print_quantum_results(result2)
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: More complex circuit
    complex_qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];
    x q[0];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    """
    
    print("3. Running 3-qubit complex circuit:")
    result3 = run_qasm_on_quantum_computer(
        qasm_string=complex_qasm,
        shots=3000,
        optimization_level=2
    )
    print_quantum_results(result3)