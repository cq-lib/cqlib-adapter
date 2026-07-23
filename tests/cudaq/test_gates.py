from __future__ import annotations

import pytest

from cqlib_adapter.cudaq.gates import (
    SUPPORTED_OPENQASM_GATES,
    cqlib_gate_for_openqasm,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("H", "H"), ("cx", "CX"), ("u3", "U"), ("measure", "MEASURE")],
)
def test_openqasm_gate_mapping_normalizes_cudaq_output(source: str, expected: str) -> None:
    assert cqlib_gate_for_openqasm(source) == expected
    assert source.lower() in SUPPORTED_OPENQASM_GATES


def test_unknown_openqasm_gate_is_not_silently_accepted() -> None:
    with pytest.raises(KeyError, match="unsupported CUDA-Q OpenQASM gate"):
        cqlib_gate_for_openqasm("opaque-custom")
