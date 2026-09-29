# Cirq tests

The suite uses Cirq `1.7.0` and the real Rust-compiled `cqlib` extension installed from PyPI.

- `test_gates.py`: QCIS gate matrices, inverses, parameters, factory errors and GPHASE.
- `test_converter.py`: Circuit conversion, measurement keys/invert masks, native lowering, layout, topology routing, reverse conversion and invalid circuits.
- `test_result.py`: canonical bit order to standard `cirq.ResultDict`, multiple keys, histograms and malformed metadata.
- `test_sampler_device.py`: DeviceMetadata, topology validation, Sampler, sweeps, DataFrame, task metadata and mock cloud.
- `test_topology_example.py`: no-network validation of the real-cloud topology example.
- `../integration/test_cirq_cqlib_runtime.py`: converter and compiler proofs against the real native extension.
- `../cloud/test_cirq_tianyan_live.py`: explicitly enabled real Tianyan submission, skipped by default.

Run:

```powershell
python -m pytest tests\cirq -q
python -m pytest tests\integration\test_cirq_cqlib_runtime.py -q
```
