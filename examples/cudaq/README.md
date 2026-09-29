# CUDA-Q examples

These examples target `cudaq>=0.15,<0.17` on Linux. Windows users should run
them in WSL2. Run every command from the repository root.

- `01_conversion.py`: CUDA-Q kernel -> Quake MLIR -> cqlib Circuit -> native
  QCIS; also proves automatic final measurement.
- `02_mock_closed_loop.py`: no-network synchronous/asynchronous job-contract
  test with preconfigured counts; it does not prove circuit semantics.
- `03_tianyan_cloud.py`: explicitly authorized real Tianyan submission; reads the key only from the current process environment.
- `031_tianyan_topology.py`: choose and validate one physical three-qubit path before optional submission.

The production converter traverses CUDA-Q's Quake MLIR objects directly and
never uses OpenQASM as a fallback. `cudaq_to_openqasm()` remains available only
as an explicit diagnostic export.

Offline commands:

```bash
python examples/cudaq/01_conversion.py
python examples/cudaq/02_mock_closed_loop.py
```

The following commands can create external tasks only after a key is supplied;
both check device availability before submission:

```bash
python examples/cudaq/03_tianyan_cloud.py
python examples/cudaq/031_tianyan_topology.py
```
