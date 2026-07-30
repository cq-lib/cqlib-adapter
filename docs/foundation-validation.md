# Project and common-core validation record

This record describes how the project skeleton and shared adapter core are
validated against the locally compiled Python extensions. It deliberately tests
the public Python API exposed by the native modules and does not test or modify
the Rust implementation.

## Validated environment

- Python: the `cqlib-adapter-dev` Conda environment (Python 3.11, 64-bit)
- `cqlib==0.1.0`: editable install from the sibling `cqlib/crates/binding-python` checkout
- `cqlib-tianyan==0.1.0`: editable install from
  the sibling `cqlib-tianyan/crates/binding-python` checkout
- `cqlib-adapter==2.0.0.dev0`: editable install from this repository

The integration tests verify that `cqlib._native` and
`cqlib_tianyan._cqlib_tianyan` are loaded from compiled `.pyd` files. This
prevents a fake module from accidentally making the native compatibility suite
pass.

## Project checks

- Independent extras exist for Qiskit, Cirq, PennyLane and CUDA-Q; `all` is
  exactly their union.
- Base dependencies target the compiled product line:
  `cqlib==0.1.0` and `cqlib-tianyan==0.1.0`.
- The development environment includes Python 3.11, Maturin, NumPy, build,
  Twine, Ruff, Mypy, pytest and coverage support.
- The source distribution contains documentation, examples and the complete
  test suite; the wheel contains only runtime package files.
- Import isolation tests ensure that installing one framework extra does not
  force imports from another framework.

## Common-core checks

- Native `Circuit`, `Instruction`, `CompileMode`, `Compiler`, `Device`,
  `Topology`, `Layout`, `ResourcePolicy`, QCIS parser and QCIS serializer are
  exercised directly.
- Logical gates are lowered to the native QCIS basis.
- `measure_bits` serialization expansion is handled correctly: QCIS parsing
  produces individual `measure_bit` instructions and measurement bindings are
  created from that executable circuit.
- Device topology validation accepts a physical edge in either direction,
  matching the compiler's routing semantics.
- Logical-to-physical layouts, sparse physical qubit IDs, invalid qubits,
  parser failures and compile failures are covered.
- Native `ExecutionResult` objects are converted to canonical counts,
  probabilities and samples.
- A local no-network closed loop covers compile -> QCIS -> platform submit ->
  wait -> canonical result.
- The public Python signatures of `TianyanPlatform`, `Backend` and `TaskHandle`
  are checked, and connector login/error normalization is tested without making
  a cloud request.

## Reproduce locally

Run the following commands in PowerShell:

```powershell
conda activate cqlib-adapter-dev
# Run the remaining commands from the cqlib-adapter repository root.
python -c "import cqlib._native as n, cqlib_tianyan._cqlib_tianyan as t; print(n.__file__); print(t.__file__)"
python -m pip check
python -m pytest -m "not cloud and not cudaq" -q
python -m pytest --cov=cqlib_adapter --cov-report=term-missing -m "not cloud and not cudaq" -q
python -m ruff check .
python -m ruff format --check .
python -m mypy cqlib_adapter
python -m build
python -m twine check dist\*
```

No cloud credentials are required for these checks. Live Tianyan submission belongs to
the later connector milestone and must be run explicitly with an authorized
account and device.

## Upstream test-suite observation

The adapter suite targets the behavior of the installed native `.pyd` modules.
Some tests currently stored in the `cqlib` source repository describe an older
Python wrapper shape (for example, enum representation, nested instruction
access and exception types) and therefore do not all pass against the current
compiled extension. This is an upstream source-test/binary mismatch, not an
adapter workaround: compatibility tests use the actual exposed API.
