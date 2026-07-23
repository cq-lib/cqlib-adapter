from __future__ import annotations

import logging
import runpy
from pathlib import Path
from typing import Any, cast

import pytest

cudaq = pytest.importorskip("cudaq")
pytest.importorskip("cqlib")

pytestmark = pytest.mark.cudaq


def test_scaling_statevector_smoke_case() -> None:
    example = (
        Path(__file__).resolve().parents[2] / "examples" / "cudaq" / "07_scaling_statevector.py"
    )
    namespace = runpy.run_path(str(example))
    logger = logging.getLogger("test.cudaq.scaling")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    cudaq.set_target("qpp-cpu", precision="fp64")

    execute_case = cast(Any, namespace["execute_case"])
    case = execute_case(2, 5, 2026, logger, False)

    assert case.passed
    assert case.fidelity >= 1.0 - 1e-10
    assert case.max_aligned_amplitude_error <= 1e-6
    assert case.qcis_lines > 0
