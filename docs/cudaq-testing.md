# CUDA-Q closed-loop validation

## Supported environment

The CUDA-Q adapter targets `cudaq>=0.15,<0.16` on Linux. The verified
environment uses Python 3.12 and `qpp-cpu`. Native Windows is unsupported by
CUDA-Q; Windows users should run these checks in WSL2.

The production path is:

```text
CUDA-Q kernel or PyKernel builder
  -> concrete Quake MLIR object
  -> strict static traversal
  -> cqlib.Circuit
  -> cqlib compile/layout/routing
  -> QCIS
  -> local cqlib simulator or Tianyan task
  -> CanonicalResult
  -> CudaQSampleResult
```

It does not call `cudaq.translate`, parse QASM, silently skip an operation, or
fall back to QASM when direct conversion fails. `cudaq_to_openqasm()` is a
separate, opt-in diagnostic export.

The supported static subset includes fixed-width single or multiple
`qalloc`/`qvector` allocations, parameterized decorators, simple scalar
parameter builders and arithmetic, statically evaluable loops, supported
unitary gates, terminal `mx`/`my`/`mz`, and automatic full `mz` when no explicit
measurement exists. Dynamic allocation/control, mid-circuit feedback,
unsupported kernel calls, state initialization, noise, `exp_pauli`, and unknown
operations raise `AdapterConversionError` with the operation named.

## Prepare WSL

From PowerShell:

```powershell
wsl.exe --list --verbose
wsl.exe -d Ubuntu-24.04
```

Then in the Ubuntu shell:

```bash
source /path/to/cqlib-adapter-cudaq/bin/activate
cd /path/to/cqlib-adapter
python -c 'import cudaq, cqlib, cqlib_tianyan, cqlib_adapter; cudaq.set_target("qpp-cpu"); print(cudaq.__version__, cudaq.get_target()); print(cqlib_adapter.__file__)'
```

Before proceeding, confirm that `cqlib_adapter.__file__` points at this
checkout and that `cudaq.get_target()` reports `qpp-cpu`.

## Module-by-module tests

Run from the repository root in the activated WSL environment:

```bash
# Direct Quake conversion, parameters, multi-qalloc, loops and boundaries
python -m pytest tests/cudaq/test_converter.py -q
python examples/cudaq/01_conversion.py

# Result API and canonical 110 -> CUDA-Q 011 bit-order projection
python -m pytest tests/cudaq/test_result.py -q
python examples/cudaq/05_bit_order_011.py

# Tianyan target/device information and diagnostic gate mapping
python -m pytest tests/cudaq/test_target.py tests/cudaq/test_gates.py -q

# Sync/async jobs, IDs, status, wait, mock transport and local simulator
python -m pytest tests/cudaq/test_execution.py -q
python examples/cudaq/02_mock_closed_loop.py

# Algorithm semantics, exact-state scaling and native basis measurements
python examples/cudaq/04_grover_simulator.py
python examples/cudaq/07_scaling_statevector.py --smoke
python examples/cudaq/08_basis_measurement.py

# Native integration and the complete offline CUDA-Q suite
python -m pytest tests/integration/test_cudaq_cqlib_runtime.py -q
python -m pytest tests/cudaq tests/integration/test_cudaq_cqlib_runtime.py -q
```

The conversion tests monkeypatch `cudaq.translate` to fail and still require
direct conversion to succeed. Key example results are `auto_measure_all=True`,
mock task ID `mock-cudaq-task-1`, local task ID `local-cudaq-task-1`, Grover
counts `{'11': shots}`, bit-order counts `{'011': shots}`, and basis counts
`{'00': shots}`.

The full scaling matrix is intentionally heavier:

```bash
python examples/cudaq/07_scaling_statevector.py
```

It compares CUDA-Q and cqlib statevectors up to global phase and writes its
generated report under ignored `test-output/`.

## Optional real-device tests

No task is created by the offline commands above. Only after explicit
authorization, read the key without terminal echo in the WSL Bash shell:

```bash
IFS= read -rsp 'Tianyan API key: ' TIANYAN_API_KEY
printf '\n'
export TIANYAN_API_KEY
export CQLIB_RUN_CLOUD=1
export TIANYAN_DEVICE='tianyan176'
export TIANYAN_TEST_SHOTS=100
python -m pytest tests/cloud/test_cudaq_tianyan_live.py -q -s -p no:cacheprovider
```

The manual cloud and topology examples also read only the current process
environment:

```bash
python examples/cudaq/03_tianyan_cloud.py
python examples/cudaq/031_tianyan_topology.py
```

Clear credentials afterward:

```bash
unset TIANYAN_API_KEY TIANYAN_DEVICE CQLIB_RUN_CLOUD
unset TIANYAN_TEST_SHOTS TIANYAN_TEST_TIMEOUT TIANYAN_POLL_INTERVAL
```
