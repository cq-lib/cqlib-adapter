# CUDA-Q closed-loop validation

## Supported environment

The CUDA-Q adapter targets Python 3.12 and `cudaq>=0.15,<0.16` on Linux. The verified local
target is `qpp-cpu`. Native Windows is intentionally unsupported; use WSL2.

```text
CUDA-Q kernel
  -> synthesize decorated kernel arguments
  -> cudaq.translate(..., format="openqasm2")
  -> add full final measurement when mz is absent
  -> cqlib.ir.qasm2.loads
  -> cqlib compile/layout/routing
  -> QCIS
  -> local cqlib simulator or Tianyan task
  -> CanonicalResult
  -> CudaQSampleResult
```

CUDA-Q 0.15 parameterized `PyKernel` builders cannot be specialized to
OpenQASM 2 and raise an explicit error. Use a typed `@cudaq.kernel` and pass
arguments to `sample`, `submit`, `cudaq_to_cqlib`, or `compile_cudaq_kernel`.
The adapter also limits translation to one `qalloc`/`qvector` because CUDA-Q 0.15 cannot
reliably export multiple allocations to OpenQASM 2.

## Prepare a WSL/Linux environment

From PowerShell, first confirm that the exact distribution is registered:

```powershell
wsl.exe --list --verbose
```

Enter WSL, activate a CUDA-Q virtual environment, and change to your checkout.
The variable below is only a portable default and may be overridden:

```powershell
wsl.exe -d Ubuntu-24.04
```

```bash
export CUDAQ_VENV="${CUDAQ_VENV:-$HOME/venvs/cqlib-adapter-cudaq}"
source "$CUDAQ_VENV/bin/activate"
cd /path/to/cqlib-adapter
python -c 'import cudaq, cqlib, cqlib_tianyan, cqlib_adapter; cudaq.set_target("qpp-cpu"); print(cudaq.__version__, cudaq.get_target()); print(cqlib_adapter.__file__)'
```

Confirm that `cqlib_adapter.__file__` points to the intended checkout before
testing. Install the two native `0.1.0` dependencies as described in the root
README before installing the adapter extra.

## Module-by-module manual tests

Run inside WSL from the repository root after activating the CUDA-Q environment:

```bash
# 1. Kernel/OpenQASM/cqlib/QCIS conversion and boundaries
python -m pytest tests/cudaq/test_converter.py -q
python examples/cudaq/01_conversion.py

# 2. Result behavior and the canonical 110 -> CUDA-Q 011 projection
python -m pytest tests/cudaq/test_result.py -q
python examples/cudaq/05_bit_order_011.py

# 3. Tianyan target/device metadata and documented gates
python -m pytest tests/cudaq/test_target.py tests/cudaq/test_gates.py -q

# 4. Sync/async jobs, IDs, status, wait, mock cloud and local simulator
python -m pytest tests/cudaq/test_execution.py -q
python examples/cudaq/02_mock_closed_loop.py

# 5. Real cqlib simulator algorithm path
python examples/cudaq/04_grover_simulator.py

# 6. Native integration and the complete offline CUDA-Q suite
python -m pytest tests/integration/test_cudaq_cqlib_runtime.py -q
python -m pytest tests/cudaq tests/integration/test_cudaq_cqlib_runtime.py -q

# 7. Exact statevector scaling and basis measurement
python examples/cudaq/07_scaling_statevector.py --smoke
python examples/cudaq/08_basis_measurement.py
```

Expected key checks are `auto_measure_all=True` in conversion, mock task ID
`mock-cudaq-task-1`, local task ID `local-cudaq-task-1`, Grover counts
`{'11': shots}`, and the bit-order example counts `{'011': shots}`.

## Explicit real-device tests

No real task is created unless you explicitly supply credentials and enable
the cloud test. Read the key without terminal echo so it is not stored in
shell history; press Enter after pasting it:

```bash
IFS= read -rsp 'Tianyan API key: ' TIANYAN_API_KEY
printf '\n'
export TIANYAN_API_KEY
export CQLIB_RUN_CLOUD=1
export TIANYAN_DEVICE='tianyan176'
export TIANYAN_TEST_SHOTS=100
python -m pytest tests/cloud/test_cudaq_tianyan_live.py -q -s -p no:cacheprovider
```

The manual real-device and topology examples keep the committed key field blank
and read `TIANYAN_API_KEY` only from the current process environment:

```bash
python examples/cudaq/03_tianyan_cloud.py
python examples/cudaq/031_tianyan_topology.py
```

The topology example compiles and validates the selected physical path and
measurement mapping before checking availability or creating a task.

Always clear the current shell after a live test:

```bash
unset TIANYAN_API_KEY TIANYAN_DEVICE CQLIB_RUN_CLOUD
unset TIANYAN_TEST_SHOTS TIANYAN_TEST_TIMEOUT TIANYAN_POLL_INTERVAL
```
