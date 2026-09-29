# Testing strategy

The default tests are offline and deterministic. They cover packaging boundaries, optional dependency isolation, shared data contracts, the real Rust-compiled Python APIs, compiler invariants, QCIS round-trips, device normalization, result bit ordering, and a fake network transport. No test marked `cloud` runs implicitly and credentials must never be printed.

## Test markers

- `unit`: fast tests for adapter-owned validation and error behavior.
- `integration`: local behavior across `cqlib-adapter`, compiled `cqlib`, and compiled `cqlib-tianyan`; no cloud credentials required.
- `cloud`: real Tianyan credentials, network access, and external task creation.
- `slow`: intentionally long-running tests.
- `qiskit`, `cirq`, `pennylane`, `cudaq`: tests requiring the corresponding optional framework.

## Required native Python packages

`environment-dev.yml` creates the Python, optional-framework, and tooling environment. The integration suite additionally loads the native `cqlib` and `cqlib-tianyan` bindings, which are published on PyPI with prebuilt wheels for the supported platforms. They are declared as project dependencies in `pyproject.toml`, so installing the adapter pulls them automatically:

```bash
# Windows/macOS: Qiskit, Cirq and PennyLane plus quality tools.
python -m pip install -e ".[dev]"

# Linux/WSL: add CUDA-Q for the complete four-framework suite.
python -m pip install -e ".[dev,cudaq]"
```

Verify both versions and native extension files before testing:

```bash
python -c "from importlib.metadata import version; import cqlib._native, cqlib_tianyan._cqlib_tianyan; print(version('cqlib'), cqlib._native.__file__); print(version('cqlib-tianyan'), cqlib_tianyan._cqlib_tianyan.__file__)"
```

Both versions must satisfy the compatible ranges declared in `pyproject.toml`; the native files must be `.pyd` on Windows or a Python extension `.so` on Linux/macOS.

## Shared-core coverage

The suite exercises the real public Python interfaces relevant to the adapter:

- `cqlib.Circuit`, native QCIS gates, `measure_bit` and `measure_bits`;
- `cqlib.compile.compile`, `CompileMode`, `Layout`, `ResourcePolicy`, routing, target basis, and workflow reports;
- `cqlib.ir.qcis.dumps/loads`, including the expansion of `measure_bits` during round-trip;
- `cqlib.device.Device`, `Topology`, `Instruction`, `Qubit`, and `ExecutionResult`;
- `cqlib_tianyan.TianyanConfig`, `CalibrationMode`, `TianyanPlatform`, `TianyanBackend`, and `TaskHandle` Python signatures;
- a real-cqlib compile -> QCIS -> submit -> wait -> canonical-result path with only the external network transport replaced by a deterministic fake;
- invalid indices, basis violations, unsupported topology, measurement mismatches, task ordering, timeouts, duplicate or unknown task IDs, malformed counts/probabilities, and failed states.

Fakes remain only for failure injection and cloud transport determinism. They mirror the native construction IR names (`measure_bit`/`measure_bits`) and native `Instruction` object shape; they are not the sole evidence for cqlib compatibility.

## Commands

```bash
# Windows: CUDA-Q is verified separately in WSL/Linux.
python -m pytest -m "not cloud and not cudaq"
python -m pytest tests/integration -m "not cudaq"
python -m pytest --cov=cqlib_adapter --cov-config=coverage-windows.ini --cov-report=term-missing -m "not cloud and not cudaq"

# Linux/WSL after installing .[dev,cudaq].
python -m pytest -m "not cloud"
python -m pytest --cov=cqlib_adapter --cov-report=term-missing -m "not cloud"
python -m ruff check .
python -m ruff format --check .
python -m mypy cqlib_adapter
python -m pip check
```

`coverage-windows.ini` intentionally omits only `cqlib_adapter.cudaq`: CUDA-Q is not
available on Windows and is verified by the Linux/WSL coverage command. Test counts
may grow as tests are added, so command exit status, explicit expected skips and
assertions are authoritative. Framework-specific offline and cloud examples are
documented in `examples/README.md` and in each framework's example directory.

CUDA-Q runs as a dedicated Linux job in `.github/workflows/tests.yml`.
See `docs/cudaq-testing.md` for its module-by-module WSL commands.

Real cloud tests create external tasks. They require explicit selection and all documented environment gates; never run them as part of a release check:

```bash
python -m pytest -m cloud
```
