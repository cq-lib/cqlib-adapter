from __future__ import annotations

import subprocess
import sys

import pytest

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")


def run_clean_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def test_top_level_import_does_not_import_quantum_frameworks() -> None:
    code = (
        "import sys, cqlib_adapter; assert not "
        + repr(set(FRAMEWORKS))
        + ".intersection(sys.modules)"
    )
    result = run_clean_python(code)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_adapter_namespace_import_is_lazy(framework: str) -> None:
    code = f"import sys; import cqlib_adapter.{framework}; assert {framework!r} not in sys.modules"
    result = run_clean_python(code)
    assert result.returncode == 0, result.stderr
