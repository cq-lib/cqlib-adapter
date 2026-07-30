# This code is part of cqlib.
#
# Copyright (C) 2025-2026 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
