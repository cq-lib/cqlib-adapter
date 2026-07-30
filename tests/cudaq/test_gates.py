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

import pytest

from cqlib_adapter.cudaq.gates import (
    SUPPORTED_OPENQASM_GATES,
    cqlib_gate_for_openqasm,
)

pytestmark = pytest.mark.cudaq


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
