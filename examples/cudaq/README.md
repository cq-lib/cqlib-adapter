# CUDA-Q examples

These examples target `cudaq>=0.15,<0.16` on Linux. Windows users should run
them in WSL2. Run every command from the repository root.

- `01_conversion.py`: CUDA-Q kernel -> OpenQASM 2 -> cqlib Circuit -> native QCIS; also proves automatic final measurement.
- `02_mock_closed_loop.py`: no-network synchronous/asynchronous job-contract
  test with preconfigured counts; it does not prove circuit semantics.
- `03_tianyan_cloud.py`: explicitly authorized real Tianyan submission; reads the key only from the current process environment.
- `031_tianyan_topology.py`: choose and validate one physical three-qubit path before optional submission.
- `04_grover_simulator.py`: compare CUDA-Q `qpp-cpu` with the real cqlib simulator path.
- `05_bit_order_011.py`: prove canonical `110` is exposed in CUDA-Q q0-left order as `011`.
- `07_scaling_statevector.py`: compare CUDA-Q `get_state` with native-QCIS cqlib
  amplitudes across increasing qubit counts and logical depths.
- `08_basis_measurement.py`: deterministic explicit X-basis rotation through
  OpenQASM 2, native QCIS and the local cqlib simulator.

Offline commands:

```bash
python examples/cudaq/01_conversion.py
python examples/cudaq/02_mock_closed_loop.py
python examples/cudaq/04_grover_simulator.py
python examples/cudaq/05_bit_order_011.py
python examples/cudaq/07_scaling_statevector.py --smoke
python examples/cudaq/08_basis_measurement.py
```

The complete exact-state matrix (2/4/6/8/12/14 qubits x depth 5/10/15/20)
is intentionally heavier and writes a detailed log under `test-output/`:

```bash
python examples/cudaq/07_scaling_statevector.py
```

This example is completely offline and never reads or submits an API key. It
returns exit code 0 only when every case passes all of these checks: normalized
states, fidelity at least `0.9999999999`, phase-aligned amplitude error at most
`1e-6`, complete logical/physical layout, native `CZ` present, and no `CX` or
measurement in the simulated QCIS. The observed values and QCIS checks are
written to the printed `test-output/cudaq_scaling_statevector_*.log` path.

The following commands can create external tasks only after a key is supplied;
both check device availability before submission:

```bash
python examples/cudaq/03_tianyan_cloud.py
python examples/cudaq/031_tianyan_topology.py
```
