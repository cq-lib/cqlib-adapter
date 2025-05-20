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

Log in to the [TianYan Lab](https://qc.zdxlz.com/), retrieve your `Connection Key` from the Dashboard page, 
and replace `your_token` in the code below.

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ext import X2PGate

# Create a quantum circuit
qs = QuantumRegister(2)
cs = ClassicalRegister(2)
circuit = QuantumCircuit(qs, cs)
circuit.x(qs[1])
circuit.h(qs[0])
circuit.cx(qs[0], qs[1])
circuit.append(X2PGate(), [qs[0]])
circuit.barrier(qs)
circuit.measure(qs, cs)

circuit.draw()
```

Circuit Text Diagram:
```text
      ┌───┐     ┌─────┐ ░ ┌─┐   
q0_0: ┤ H ├──■──┤ X2p ├─░─┤M├───
      ├───┤┌─┴─┐└─────┘ ░ └╥┘┌─┐
q0_1: ┤ X ├┤ X ├────────░──╫─┤M├
      └───┘└───┘        ░  ║ └╥┘
c0: 2/═════════════════════╩══╩═
                           0  1 
```


#### 1. Backend mode
```python
from cqlib_adapter.qiskit_ext import TianYanProvider

# Initialize the TianYan provider
provider = TianYanProvider(token='your_token')

# Retrieve a specific backend (e.g., 'tianyan176-2')
backend = provider.backend('tianyan176-2')

# Run the circuit on the backend
job = backend.run([circuit], shots=3000)

# Retrieve and print the results
print(f'Job ID: {job.job_id()}')
print(f'Job Result: {job.result().get_counts()}')
```

#### 2. Sampler mode
```python
from cqlib_adapter.qiskit_ext import TianYanProvider, TianYanSampler

# Initialize the TianYan provider
provider = TianYanProvider(token='your_token')

# Retrieve a specific backend (e.g., 'tianyan24')
backend = provider.backend('tianyan24')

# Run the circuit on the backend
job = TianYanSampler(backend=backend).run([circuit], shots=3000)

# Retrieve and print the results
print(f'Job ID: {job.job_id()}')
print(f'Job Result: {job.result()}')
# c0 is the default register name
# cs = ClassicalRegister(2)
print(f'Counts: {job.result()[0].data.c0.get_counts()}')
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
