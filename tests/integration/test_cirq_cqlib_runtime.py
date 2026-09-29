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

import cirq
import pytest
from cqlib import Circuit

from cqlib_adapter.cirq import (
    cirq_to_cqlib,
    compile_cirq_circuit,
)
from cqlib_adapter.common import CompilationOptions

pytestmark = [pytest.mark.integration, pytest.mark.cirq]
NATIVE_BASIS = (
    "RZ",
    "X2P",
    "X2M",
    "Y2P",
    "Y2M",
    "XY2P",
    "XY2M",
    "CZ",
    "GPHASE",
)


def test_cirq_converter_returns_rust_backed_cqlib_circuit() -> None:
    import cqlib._native as native

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(*qubits), cirq.measure(*qubits, key="m"))

    bundle = cirq_to_cqlib(circuit)

    assert isinstance(bundle.circuit, Circuit)
    assert any(str(Path(native.__file__).resolve()).endswith(item) for item in EXTENSION_SUFFIXES)
    assert bundle.circuit.validate() is None


def test_cirq_compiler_calls_real_cqlib_compile_and_qcis() -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(*qubits), cirq.measure(*qubits, key="m"))

    artifact = compile_cirq_circuit(circuit, options=CompilationOptions(target_basis=NATIVE_BASIS))

    assert artifact.circuit.validate() is None
    assert "CZ Q0 Q1" in artifact.qcis
    assert "M Q0" in artifact.qcis
    assert "M Q1" in artifact.qcis
    assert artifact.steps
