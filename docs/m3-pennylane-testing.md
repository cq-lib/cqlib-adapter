# PennyLane closed-loop validation

## Supported versions

The PennyLane adapter is implemented against PennyLane `0.45.1` and the package extra is intentionally bounded to `pennylane>=0.45,<0.46`. It uses the current `pennylane.devices.Device` API, `CompilePipeline`, `preprocess_transforms`, `ExecutionConfig`, and `execute`; it does not use the retired legacy `QubitDevice` plugin contract.

## Closed-loop architecture

A QNode is preprocessed by PennyLane, decomposed to operations understood by the adapter, converted into a real Rust-backed `cqlib.Circuit`, compiled against the selected cqlib device/topology, serialized to QCIS, submitted through `cqlib-tianyan`, converted to canonical result data, and projected back into PennyLane measurement wire order.

The public surfaces are:

- `cqlib_adapter.pennylane.TianyanDevice` for real or mocked cloud execution;
- `cqlib_adapter.pennylane.CqlibSimulatorDevice` for offline real-cqlib simulation;
- `pennylane_to_cqlib`, `cqlib_to_pennylane`, and `compile_pennylane_circuit`;
- QCIS-native `X2P/X2M/Y2P/Y2M/XY/XY2P/XY2M/RXY/FSim` operations;
- `PennyLaneExecution` for task ID, status, submitted QCIS and cached result access.

The adapter supports finite-shot `qml.counts`, `qml.probs`, `qml.sample`, `qml.expval` and `qml.var`. Single-wire Pauli X/Y/Z observables are lowered through explicit basis rotations and projected back to PennyLane eigenvalues. Analytic QNode execution, multi-wire observables, incompatible bases on one wire, mid-circuit measurements and shot vectors are rejected explicitly. The local simulator also exposes `run_statevector()` and restores logical wire order after non-identity compilation layouts.

## Bit ordering

cqlib canonical strings are MSB-left in classical storage: the leftmost character is the highest classical index. PennyLane count strings list the requested wires from left to right. For PennyLane wires `(0,1,2)`, state `wire0=0, wire1=1, wire2=1` is stored canonically as `110` and must be returned as `011`. `05_bit_order_011.py` and the result tests assert counts, probability index 3 and sample rows `[0,1,1]` together.

## Manual validation order

```powershell
conda activate cqlib-adapter-dev
# Run the remaining commands from the cqlib-adapter repository root.
python -c "import pennylane as qml; print(qml.__version__)"
python -m pip install -e ".[pennylane]"
python -m pytest tests\pennylane\test_operations.py -q
python -m pytest tests\pennylane\test_converter.py -q
python examples\pennylane\01_conversion.py
python -m pytest tests\pennylane\test_result.py -q
python examples\pennylane\05_bit_order_011.py
python -m pytest tests\pennylane\test_device.py -q
python -m pytest tests\pennylane\test_topology_example.py -q
python examples\pennylane\02_mock_closed_loop.py
python examples\pennylane\04_grover_simulator.py
python -m pytest tests\pennylane -q
```

Only after all offline commands pass, load the key into the current process environment using the hidden-input PowerShell procedure in `docs/m2-qiskit-testing.md`, verify `DEFAULT_DEVICE`, and run `examples/pennylane/03_tianyan_cloud.py`. Never fill the committed blank key field. Then run `python examples\pennylane\031_tianyan_topology.py` to validate a selected three-qubit physical layout before submitting the topology test. The script prints device metadata before submission and skips without creating a task when the device is unavailable.

The CI-safe live test is skipped by default. Explicit authorization requires:

```powershell
$env:CQLIB_RUN_CLOUD="1"
$env:TIANYAN_DEVICE="tianyan176"
python -m pytest tests\cloud\test_pennylane_tianyan_live.py -q -s
```

Clear `TIANYAN_API_KEY` immediately after the test. Do not commit a key or put it in shell history.
