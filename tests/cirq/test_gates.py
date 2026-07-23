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

from math import pi

import cirq
import numpy as np
import pytest
import sympy

from cqlib_adapter.cirq import (
    RXYGate,
    X2MGate,
    X2PGate,
    XY2MGate,
    XY2PGate,
    XYGate,
    Y2MGate,
    Y2PGate,
    qcis_gate,
)

pytestmark = pytest.mark.cirq


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        (X2PGate(), X2MGate()),
        (Y2PGate(), Y2MGate()),
        (XY2PGate(0.3), XY2MGate(0.3)),
    ],
)
def test_half_rotation_inverse_pairs(positive: cirq.Gate, negative: cirq.Gate) -> None:
    assert positive**-1 == negative
    assert negative**-1 == positive
    np.testing.assert_allclose(
        cirq.unitary(positive) @ cirq.unitary(negative),
        np.eye(2),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "gate",
    [
        X2PGate(),
        X2MGate(),
        Y2PGate(),
        Y2MGate(),
        XYGate(0.2),
        XY2PGate(0.4),
        XY2MGate(0.4),
        RXYGate(0.2, 0.4),
        cirq.FSimGate(0.2, 0.4),
    ],
)
def test_qcis_gate_unitaries_are_unitary(gate: cirq.Gate) -> None:
    matrix = cirq.unitary(gate)
    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=1e-12)


def test_parameterized_native_gate_resolves_through_cirq_protocol() -> None:
    theta, phi = sympy.symbols("theta phi")
    gate = RXYGate(theta, phi)
    assert cirq.is_parameterized(gate)
    assert cirq.parameter_names(gate) == {"theta", "phi"}
    resolved = cirq.resolve_parameters(gate, {"theta": pi / 3, "phi": pi / 7})
    assert resolved == RXYGate(pi / 3, pi / 7)
    assert cirq.has_unitary(resolved)


def test_qcis_gate_factory_normalizes_names_and_validates_parameters() -> None:
    assert qcis_gate("x2p") == X2PGate()
    assert qcis_gate("CZ") == cirq.CZ
    assert isinstance(qcis_gate("fsim", 0.1, 0.2), cirq.FSimGate)
    assert isinstance(qcis_gate("gphase", pi / 3), cirq.GlobalPhaseGate)
    with pytest.raises(KeyError, match="unsupported QCIS gate"):
        qcis_gate("not-a-gate")
    with pytest.raises(ValueError, match="requires 2"):
        qcis_gate("rxy", 0.1)
