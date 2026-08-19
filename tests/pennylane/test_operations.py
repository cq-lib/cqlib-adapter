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

import numpy as np
import pennylane as qml
import pytest
from cqlib.circuit.gates import XY as CqlibXY

from cqlib_adapter.pennylane import (
    RXY,
    X2M,
    X2P,
    XY,
    XY2M,
    XY2P,
    Y2M,
    Y2P,
    FSim,
)

pytestmark = pytest.mark.pennylane


@pytest.mark.parametrize(
    "operation",
    [
        X2P(wires=0),
        X2M(wires=0),
        Y2P(wires=0),
        Y2M(wires=0),
        XY(0.3, wires=0),
        XY2P(0.3, wires=0),
        XY2M(0.3, wires=0),
        RXY(0.4, -0.2, wires=0),
        FSim(0.4, -0.2, wires=(0, 1)),
    ],
)
def test_qcis_operation_matrix_is_unitary(operation: object) -> None:
    matrix = np.asarray(qml.matrix(operation), dtype=complex)
    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=1e-10)


@pytest.mark.parametrize("axis", [0.0, 0.31, -0.29, pi / 2])
def test_xy_operation_matrix_matches_cqlib(axis: float) -> None:
    np.testing.assert_allclose(
        qml.matrix(XY(axis, wires=0)),
        CqlibXY.matrix([axis]),
        atol=1e-12,
    )


def test_xy_operation_adjoint_uses_opposite_axis() -> None:
    axis = 0.31
    operation = XY(axis, wires="q")
    adjoint = operation.adjoint()

    assert tuple(adjoint.data) == pytest.approx((axis + pi,))
    np.testing.assert_allclose(
        np.asarray(qml.matrix(adjoint)) @ np.asarray(qml.matrix(operation)),
        np.eye(2),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        (X2P(wires="q"), X2M(wires="q")),
        (Y2P(wires="q"), Y2M(wires="q")),
        (XY2P(0.7, wires="q"), XY2M(0.7, wires="q")),
    ],
)
def test_half_rotation_adjoint_pairs(positive: object, negative: object) -> None:
    adjoint = positive.adjoint()  # type: ignore[attr-defined]
    assert isinstance(adjoint, type(negative))
    assert adjoint.wires == negative.wires
    product = np.asarray(qml.matrix(adjoint)) @ np.asarray(qml.matrix(positive))
    np.testing.assert_allclose(product, np.eye(2), atol=1e-10)


def test_parameterized_adjoint_preserves_axis_and_negates_angle() -> None:
    rxy = RXY(0.4, 0.7, wires=2).adjoint()
    fsim = FSim(0.4, 0.7, wires=(0, 1)).adjoint()
    assert tuple(rxy.data) == pytest.approx((-0.4, 0.7))
    assert tuple(fsim.data) == pytest.approx((-0.4, -0.7))


def test_native_operations_can_run_on_default_qubit() -> None:
    device = qml.device("default.qubit", wires=2)

    @qml.qnode(device)
    def circuit() -> np.ndarray:
        X2P(wires=0)
        XY2M(0.2, wires=1)
        FSim(0.1, -0.3, wires=(0, 1))
        return qml.probs(wires=(0, 1))

    probabilities = circuit()
    assert probabilities.shape == (4,)
    assert probabilities.sum() == pytest.approx(1.0)
