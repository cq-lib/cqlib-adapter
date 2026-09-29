# Cirq examples

These examples target `cirq-core>=1.4,<2` and were verified with Cirq `1.7.0`.
Run them from the repository root.

- `01_conversion.py`: `cirq.Circuit -> cqlib.Circuit -> native QCIS`, including measurement-key metadata.
- `02_mock_closed_loop.py`: no-network Sampler/job/result contract test with
  preconfigured counts; it does not prove circuit semantics.
- `03_tianyan_cloud.py`: explicitly authorized real `tianyan176` submission; reads the key only from the current process environment.
- `031_tianyan_topology.py`: select a connected three-qubit physical path, validate Cirq-to-cqlib layout/topology, then optionally submit.

Install only the Cirq adapter:

```powershell
python -m pip install -e ".[cirq]"
```

Run the offline examples:

```powershell
python examples\cirq\01_conversion.py
python examples\cirq\02_mock_closed_loop.py
```

The following commands create external tasks only when the configured device is running:

```powershell
python examples\cirq\03_tianyan_cloud.py
python examples\cirq\031_tianyan_topology.py
```
