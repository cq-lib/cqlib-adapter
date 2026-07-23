# Cirq examples

These examples target `cirq-core>=1.7,<2` and were verified with Cirq `1.7.0`.
Run them from the repository root.

- `01_conversion.py`: `cirq.Circuit -> cqlib.Circuit -> native QCIS`, including measurement-key metadata.
- `02_mock_closed_loop.py`: no-network Sampler/job/result contract test with
  preconfigured counts; it does not prove circuit semantics.
- `03_tianyan_cloud.py`: explicitly authorized real `tianyan176` submission; reads the key only from the current process environment.
- `031_tianyan_topology.py`: select a connected three-qubit physical path, validate Cirq-to-cqlib layout/topology, then optionally submit.
- `04_grover_simulator.py`: Cirq `Simulator` versus real cqlib Statevector Grover execution and timing.
- `05_bit_order_011.py`: prove canonical `110` becomes Cirq measurement row `011` and big-endian integer `3`.
- `06_deutsch_jozsa_simulator.py`: Cirq `Simulator` versus cqlib-adapter for a balanced XOR oracle.
- 07_scaling_statevector.py: Cirq exact statevector versus native-QCIS cqlib
  Statevector over 24 qubit/depth cases, using a reversed physical layout and
  enforcing fidelity plus component-level amplitude error; no repetitions.
- `08_basis_measurement.py`: deterministic explicit X/Y-basis rotations through
  the real local cqlib path.

Install only the Cirq adapter:

```powershell
python -m pip install -e ".[cirq]"
```

Run the offline examples:

```powershell
python examples\cirq\01_conversion.py
python examples\cirq\02_mock_closed_loop.py
python examples\cirq\04_grover_simulator.py
python examples\cirq\05_bit_order_011.py
python examples\cirq\06_deutsch_jozsa_simulator.py
python examples\cirq\07_scaling_statevector.py --smoke
python examples\cirq\08_basis_measurement.py
```

The following commands create external tasks only when the configured device is running:

```powershell
python examples\cirq\03_tianyan_cloud.py
python examples\cirq\031_tianyan_topology.py
```
