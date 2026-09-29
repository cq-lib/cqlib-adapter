# PennyLane examples

These examples target PennyLane `>=0.44,<1` and run from the repository root.

- `01_conversion.py`: `QuantumScript -> cqlib Circuit -> native QCIS`, no network.
- `02_mock_closed_loop.py`: no-network QNode/Device/result contract test with
  preconfigured counts; it does not prove circuit semantics.
- `03_tianyan_cloud.py`: explicitly authorized real `tianyan176` submission; reads the key only from the current process environment.
- `031_tianyan_topology.py`: select a connected three-qubit physical path, validate PennyLane wire-to-cqlib topology/layout mapping, then optionally submit it to the real device.
- `04_grover_simulator.py`: PennyLane `default.qubit` versus real cqlib Statevector Grover execution and timing.
- `05_bit_order_011.py`: prepare `|011>` and prove canonical `110` is returned to PennyLane users as `011`.
- 06_scaling_statevector.py: PennyLane default.qubit versus native-QCIS cqlib
  Statevector over 24 wire/depth cases, using a reversed physical layout and
  enforcing fidelity plus component-level amplitude error; no shots.
- `08_basis_measurement.py`: native PennyLane Pauli X/Y observable counts,
  including basis rotation and eigenvalue conversion through local cqlib. The
  adapter supports finite-shot single-wire Pauli X/Y/Z `counts`, `sample`,
  `expval`, and `var`; incompatible bases requested on one wire are rejected.

Install only this framework adapter with:

```powershell
python -m pip install -e ".[pennylane]"
```

Run the offline sequence:

```powershell
python examples\pennylane\01_conversion.py
python examples\pennylane\02_mock_closed_loop.py
python examples\pennylane\04_grover_simulator.py
python examples\pennylane\05_bit_order_011.py
python examples\pennylane\06_scaling_statevector.py --smoke
python examples\pennylane\08_basis_measurement.py
```

The real-cloud examples create external tasks and should be run only after checking the selected device and shots:

```powershell
python examples\pennylane\03_tianyan_cloud.py
python examples\pennylane\031_tianyan_topology.py
```
