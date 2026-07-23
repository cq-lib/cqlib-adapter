# Cirq tests

The suite uses Cirq `1.7.0` and the real Rust-compiled `cqlib==0.1.0` extension.

- `test_gates.py`: QCIS gate matrices, inverses, parameters, factory errors and GPHASE.
- `test_converter.py`: Circuit conversion, measurement keys/invert masks, native lowering, layout, topology routing, reverse conversion and invalid circuits.
- `test_result.py`: canonical bit order to standard `cirq.ResultDict`, multiple keys, histograms and malformed metadata.
- `test_sampler_device.py`: DeviceMetadata, topology validation, Sampler, sweeps, DataFrame, task metadata, mock cloud, local 011 and Grover.
- `test_topology_example.py`: no-network validation of the real-cloud topology example.
- `../integration/test_cirq_cqlib_runtime.py`: direct native-extension and cqlib Statevector proof.
- `../cloud/test_cirq_tianyan_live.py`: explicitly enabled real Tianyan submission, skipped by default.

Run:

```powershell
python -m pytest tests\cirq -q
python -m pytest tests\integration\test_cirq_cqlib_runtime.py -q
```
