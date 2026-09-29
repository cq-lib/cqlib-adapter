# cqlib-adapter

[Chinese version](README_CN.md)

`cqlib-adapter` connects Qiskit, Cirq, PennyLane, and CUDA-Q programs to the current `cqlib`, `cqlib-tianyan`, and Tianyan Quantum Cloud platform.

The project provides shared compilation, QCIS, device, job, and canonical-result infrastructure, plus Qiskit, Cirq, PennyLane, and CUDA-Q adapters. Each adapter has conversion, mock-cloud, and explicitly opt-in real-device coverage. See [docs/testing.md](docs/testing.md) for test commands and [docs/cudaq-testing.md](docs/cudaq-testing.md) for CUDA-Q validation.

## Requirements

- Python 3.11 or later; CI tests the floor and latest (3.11 and 3.14) on Linux, Windows, and macOS.
- The current `cqlib` and `cqlib-tianyan` native bindings, installed automatically from PyPI as project dependencies; the compatible ranges are declared in [pyproject.toml](pyproject.toml).

## Install an adapter

```bash
pip install "cqlib-adapter[qiskit]"
pip install "cqlib-adapter[cirq]"
pip install "cqlib-adapter[pennylane]"
pip install "cqlib-adapter[cudaq]"
pip install "cqlib-adapter[qiskit,cirq]"
pip install "cqlib-adapter[all]"
```

The framework extras are independent. The base package does not import a quantum framework until its adapter is used, so a missing optional framework does not block another adapter.

CUDA-Q supports Linux and Apple Silicon macOS; use WSL2 on Windows. On Windows, `[cudaq]` and `[all]` intentionally do not install CUDA-Q, and the adapter gives a platform-specific installation message when used.

## Minimal offline examples

Install `cqlib`, `cqlib-tianyan`, and the required framework extra first; pip pulls the native bindings from PyPI automatically when the adapter is installed. Run these commands from the repository root. They neither read an API key nor create a cloud task.

| Adapter | Conversion | Mock closed loop |
|---|---|---|
| Qiskit | `python examples/qiskit/01_conversion.py` | `python examples/qiskit/02_mock_closed_loop.py` |
| PennyLane | `python examples/pennylane/01_conversion.py` | `python examples/pennylane/02_mock_closed_loop.py` |
| Cirq | `python examples/cirq/01_conversion.py` | `python examples/cirq/02_mock_closed_loop.py` |
| CUDA-Q (Linux/WSL2) | `python examples/cudaq/01_conversion.py` | `python examples/cudaq/02_mock_closed_loop.py` |

`01_conversion.py` shows the smallest framework-to-cqlib/QCIS path. `02_mock_closed_loop.py` exercises the full Backend/Sampler/Device submission path against a mock cloud transport with preconfigured counts; it validates transport, task, and result-shape contracts rather than quantum semantics. Real-device examples are described in [examples/README.md](examples/README.md) and the framework-specific example directories. `03_tianyan_cloud.py` and `031_tianyan_topology.py` create external tasks only after explicit operator authorization and credentials are provided.

### CUDA-Q direct conversion

The production CUDA-Q path reads a kernel or builder's Quake MLIR directly and constructs a `cqlib.Circuit`. It does not depend on OpenQASM 2 and never falls back to QASM after an unsupported operation. `cudaq_to_openqasm()` is an opt-in diagnostic exporter only. Supported static programs include fixed-width multiple `qalloc` allocations, parameterized decorators, simple scalar builder parameters, statically evaluable loops, and terminal `mx`/`my`/`mz` measurements. When no measurement is explicit, the adapter adds full `mz` measurement. Dynamic circuit structures fail with a clear conversion error instead of being guessed or skipped.

### PennyLane basis measurements

For finite shots, the PennyLane adapter supports single-wire Pauli X/Y/Z observables with `qml.counts`, `qml.sample`, `qml.expval`, and `qml.var`. It inserts H for X, S† then H for Y, and no rotation for Z before measurement. Canonical bits are converted to PennyLane's ±1 eigenvalues, and incompatible bases on one wire raise a clear error. Unit tests cover the basis rotations, eigenvalue conversion, and the `sample`/`expval`/`var` boundary cases.

### PennyLane authentication and run options

`TianyanDevice.login()` and `TianyanDevice.from_credentials()` separate connector authentication options from device execution options. A `timeout`, for example, cannot be forwarded to the authentication layer accidentally.

```python
device = TianyanDevice.login(
    api_key,
    "tianyan176",
    login_options={"domain": "https://platform.example"},
    timeout=120,
    poll_interval=5,
    calibration="auto",
    compilation_mode="normal",
    seed=7,
)
```

Use `credential_options={"credentials_path": ...}` with `from_credentials()` for stored credentials. Both mappings accept `domain`, `auto_refresh`, and `credentials_path`; `credential_options` also accepts `save_credentials`. Unknown options fail before authentication. Pass `save_credentials` to `login()` explicitly, not inside `login_options`.

## Local development environment

```bash
conda env create -f environment-dev.yml
conda activate cqlib-adapter-dev
```

`environment-dev.yml` creates the Python, optional-framework, and quality-tool environment. The native `cqlib` and `cqlib-tianyan` bindings are published on PyPI with prebuilt wheels for the supported platforms, so no source build is required.

Install the adapter in editable mode; pip pulls the `cqlib` and `cqlib-tianyan` dependencies from PyPI:

```bash
python -m pip install -e ".[dev]"
```

Confirm that the PyPI native extensions are loaded:

```bash
python -c "from importlib.metadata import version; import cqlib._native, cqlib_tianyan._cqlib_tianyan; print(version('cqlib'), cqlib._native.__file__); print(version('cqlib-tianyan'), cqlib_tianyan._cqlib_tianyan.__file__)"
```

The reported versions must satisfy the compatible ranges declared in [pyproject.toml](pyproject.toml).

The `dev` extra is the documented Windows development/test surface: it includes quality tools plus Qiskit, Cirq, and PennyLane. Linux/WSL uses the same extra with CUDA-Q added explicitly. This keeps declared dependencies, the environment file, documentation, and CI commands aligned:

```bash
python -m pip install -e ".[dev,cudaq]"
```

For offline import-boundary tests that do not load the native bindings, use:

```bash
python -m pip install --no-deps -e .
```

## Test and quality commands

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy cqlib_adapter
# Windows: CUDA-Q is independently verified in WSL/Linux.
python -m pytest -m "not cloud and not cudaq"
python -m pytest --cov=cqlib_adapter --cov-config=coverage-windows.ini --cov-report=term-missing -m "not cloud and not cudaq"
# Linux/WSL after installing .[dev,cudaq].
python -m pytest -m "not cloud"
python -m pytest --cov=cqlib_adapter --cov-report=term-missing -m "not cloud"
python -m build
python -m twine check dist/*
```

Cloud tests are never part of the default suite or CI. They require the explicit `cloud` marker and documented environment gates:

```bash
python -m pytest -m cloud
```

Enter credentials only through a hidden prompt in the current terminal process. Never put an API key in source files, configuration, command history, or logs, and clear credential environment variables after a real-device run. The cloud examples provide the corresponding operator instructions.

## Shared core and guarantees

`cqlib_adapter.common` provides:

- `TranslationBundle` and `TranslationMetadata`, including measurement and bit-mapping contracts.
- `CircuitCompiler`, which decomposes, maps to native gates, lays out/routes, and validates QCIS round trips, basis, topology, and measurement bindings. Measurements, barriers, and other directives do not need a coupling edge; only genuine two-qubit gates are checked, respecting symmetric versus control-target directionality.
- `NormalizedDevice`, which normalizes device names, physical qubit IDs, native gates, directed topology, state, pricing, and availability.
- `TianyanConnector`, for authentication, credential restoration, device discovery, compilation, calibration selection, and task submission.
- `AdapterJob`, for task IDs, non-blocking status, waiting, timeouts, batch result ordering, and result caching. `timeout` and `poll_interval` must be finite positive numbers; `NaN`, infinity, booleans, and non-numeric values fail before submission or waiting.
- `ResultConverter`, which reads Tianyan results in cqlib's little-endian convention and restores framework classical-bit order for counts, probabilities, and samples.

The submission layer submits circuits individually so task IDs and compilation metadata remain one-to-one. If a later submission fails, the error preserves IDs already created for recovery and inspection.

## Source-distribution boundary

The source distribution is deliberately minimal. It contains package sources, `py.typed`, the license, and base installation/import smoke tests. Documentation, examples, and the complete test suite remain in the Git checkout and are verified by CI.

```bash
python -m build
python -m twine check dist/*
```

Before committing or publishing, remove `.coverage`, caches, logs, `build/`, `dist/`, and `*.egg-info/`. They are reproducible and ignored by Git. See [docs/release-checklist.md](docs/release-checklist.md) for the release checklist.

## Project boundaries

- `cqlib_adapter.common`: shared conversion, compilation, device, job, and result infrastructure.
- `cqlib_adapter.qiskit`: Qiskit-facing API.
- `cqlib_adapter.cirq`: Cirq-facing API.
- `cqlib_adapter.pennylane`: PennyLane Device API.
- `cqlib_adapter.cudaq`: CUDA-Q target/execution API.

Authentication, HTTP transport, and Tianyan result parsing remain in the shared layer and `cqlib-tianyan`; framework adapters do not duplicate them.

## Test principles

- Tests, fixtures, and CI configuration belong in Git.
- `.gitignore` excludes only reproducible artifacts, real credentials, and cloud-task output.
- Each feature has success and boundary tests.
- Mock transport tests cover cloud behavior without a key; real-device tests are separately opt-in.

See [docs/compatibility.md](docs/compatibility.md) and [docs/architecture.md](docs/architecture.md) for details.
