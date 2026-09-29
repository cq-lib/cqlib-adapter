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

from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pennylane as qml
import pytest
from cqlib import Circuit

from cqlib_adapter.pennylane import (
    compile_pennylane_circuit,
    pennylane_to_cqlib,
)

pytestmark = [pytest.mark.integration, pytest.mark.pennylane]


def test_pennylane_converter_returns_rust_backed_cqlib_circuit() -> None:
    import cqlib._native as native

    tape = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.CNOT((0, 1))],
        [qml.counts(wires=(0, 1))],
        shots=20,
    )
    bundle = pennylane_to_cqlib(tape)
    assert isinstance(bundle.circuit, Circuit)
    assert any(str(Path(native.__file__).resolve()).endswith(item) for item in EXTENSION_SUFFIXES)
    assert bundle.circuit.validate() is None


def test_pennylane_compiler_calls_real_cqlib_compile_and_qcis() -> None:
    tape = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.CNOT((0, 1))],
        [qml.counts(wires=(0, 1))],
        shots=20,
    )
    artifact = compile_pennylane_circuit(tape)
    assert artifact.circuit.validate() is None
    assert "M Q0" in artifact.qcis
    assert "M Q1" in artifact.qcis
    assert artifact.steps
