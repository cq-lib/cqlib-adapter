# CUDA-Q tests

The tests cover OpenQASM conversion, automatic measurement, parameterized
decorator kernels, explicit PyKernel-builder limitations, target metadata,
canonical/CUDA-Q bit order, synchronous and asynchronous jobs, mock Tianyan,
the local cqlib simulator, Grover, and validation boundaries.

Run on Linux with CUDA-Q 0.15 installed:

```bash
python -m pytest tests/cudaq -q
python -m pytest tests/integration/test_cudaq_cqlib_runtime.py -q
```

The real cloud test is opt-in and never runs from the commands above.
