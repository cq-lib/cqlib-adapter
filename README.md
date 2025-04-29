# Cqlib Adapter

## Installation

Install the package using pip:

```bash
pip install cqlib-adapter
```

## 1. Qiskit Ext

This project provides a Qiskit adapter for the TianYan quantum computing platform. It includes custom quantum gates and
integrates with the TianYan backend to enable seamless execution of quantum circuits.

### Features

- **Custom Quantum Gates**: Adds custom gates like `X2P`, `X2M`, `Y2P`, `Y2M`, `XY2P`, and `XY2M` to Qiskit.
- **TianYan Backend Integration**: Supports execution of quantum circuits on TianYan quantum computers and simulators.
- **Transpilation**: Automatically transpiles Qiskit circuits to be compatible with TianYan backends.

### QCIS Gates

[QCIS Instruction Manual](https://qc.zdxlz.com/learn/#/resource/informationSpace?lang=zh&cId=/mkdocs/zh/appendix/QCIS_instruction_set.html)

The following QCIS gates are added to Qiskit:

- **X2P**: Positive X rotation by π/2.
- **X2M**: Negative X rotation by π/2.
- **Y2P**: Positive Y rotation by π/2.
- **Y2M**: Negative Y rotation by π/2.
- **XY2P**: Positive XY rotation by a parameterized angle.
- **XY2M**: Negative XY rotation by a parameterized angle.

### Usage Example

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from cqlib_adapter.qiskit_ext import TianYanProvider

# Initialize the TianYan provider
provider = TianYanProvider(token='your_token')

# Retrieve a specific backend (e.g., 'tianyan24')
backend = provider.backend('tianyan24')

# Create a quantum circuit
qs = QuantumRegister(2)
cs = ClassicalRegister(2)
circuit = QuantumCircuit(qs, cs)
circuit.x(qs[1])
circuit.h(qs[0])
circuit.cx(qs[0], qs[1])
circuit.barrier(qs)
circuit.measure(qs, cs)

# Transpile the circuit for the backend
transpiled_qc = transpile(circuit, backend=backend)

# Run the circuit on the backend
job = backend.run([transpiled_qc], shots=3000, readout_calibration=True)

# Retrieve and print the results
print(f'Job ID: {job.job_id()}')
print(f'Job Result: {job.result().get_counts()}')
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
