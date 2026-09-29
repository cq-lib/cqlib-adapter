# CUDA-Q tests

The tests cover direct Quake MLIR conversion, automatic measurement,
parameterized decorator and builder kernels, fixed multi-allocation mapping,
static loops, native `mx`/`my` basis measurements, explicit unsupported-program
errors, target metadata, canonical/CUDA-Q bit order, synchronous and
asynchronous jobs, mock Tianyan, and validation boundaries. A guard test replaces `cudaq.translate` with a function
that fails, proving the production converter does not use the diagnostic QASM
exporter.

Run on Linux with a supported CUDA-Q release (`cudaq>=0.15,<0.17`, Python 3.11-3.13) installed:

```bash
python -m pytest tests/cudaq -q
python -m pytest tests/integration/test_cudaq_cqlib_runtime.py -q
```

The real cloud test is opt-in and never runs from the commands above.
