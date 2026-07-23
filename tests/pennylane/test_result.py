from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from cqlib_adapter.common import AdapterResultError, CanonicalResult
from cqlib_adapter.pennylane import canonical_to_pennylane_result

pytestmark = pytest.mark.pennylane


def canonical_011(shots: int = 10) -> CanonicalResult:
    # Canonical storage is c2-c1-c0, while PennyLane wires are requested q0-q1-q2.
    return CanonicalResult("task", shots, {"110": shots}, {"110": 1.0}, 3)


def test_counts_reverse_canonical_storage_into_pennylane_wire_order() -> None:
    result = canonical_to_pennylane_result(
        canonical_011(),
        [qml.counts(wires=(0, 1, 2))],
        wire_order=(0, 1, 2),
        active_wires=(0, 1, 2),
    )
    assert result == {"011": 10}


def test_probs_and_samples_use_requested_wire_order() -> None:
    probabilities, samples = canonical_to_pennylane_result(
        canonical_011(4),
        [qml.probs(wires=(2, 0)), qml.sample(wires=(1, 2))],
        wire_order=(0, 1, 2),
        active_wires=(0, 1, 2),
    )
    np.testing.assert_array_equal(probabilities, [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(samples, np.asarray([[1, 1]] * 4, dtype=np.int8))


def test_empty_measurement_wires_mean_active_tape_wires() -> None:
    result = canonical_to_pennylane_result(
        canonical_011(),
        [qml.counts()],
        wire_order=(0, 1, 2),
        active_wires=(1, 2),
    )
    assert result == {"11": 10}


def test_counts_all_outcomes_includes_zero_entries() -> None:
    result = canonical_to_pennylane_result(
        canonical_011(),
        [qml.counts(wires=(0, 1), all_outcomes=True)],
        wire_order=(0, 1, 2),
        active_wires=(0, 1),
    )
    assert result == {"00": 0, "01": 10, "10": 0, "11": 0}


def test_pauli_observable_results_use_eigenvalues() -> None:
    canonical = CanonicalResult(
        "task",
        4,
        {"0": 3, "1": 1},
        {"0": 0.75, "1": 0.25},
        1,
    )

    counts, probabilities, samples, expectation, variance = canonical_to_pennylane_result(
        canonical,
        [
            qml.counts(qml.X(0), all_outcomes=True),
            qml.probs(op=qml.X(0)),
            qml.sample(qml.X(0)),
            qml.expval(qml.X(0)),
            qml.var(qml.X(0)),
        ],
        wire_order=(0,),
        active_wires=(0,),
    )

    assert counts == {1.0: 3, -1.0: 1}
    np.testing.assert_array_equal(probabilities, [0.75, 0.25])
    unique, frequencies = np.unique(samples, return_counts=True)
    assert dict(zip(unique.tolist(), frequencies.tolist(), strict=True)) == {-1.0: 1, 1.0: 3}
    assert expectation == pytest.approx(0.5)
    assert variance == pytest.approx(0.75)


def test_width_unknown_wire_and_unsupported_measurement_are_rejected() -> None:
    with pytest.raises(AdapterResultError, match="width"):
        canonical_to_pennylane_result(
            CanonicalResult("t", 1, {"0": 1}, {"0": 1.0}, 1),
            [qml.counts(wires=0)],
            wire_order=(0, 1),
            active_wires=(0,),
        )
    with pytest.raises(AdapterResultError, match="unknown wire"):
        canonical_to_pennylane_result(
            canonical_011(),
            [qml.counts(wires="missing")],
            wire_order=(0, 1, 2),
            active_wires=(0, 1, 2),
        )
    with pytest.raises(AdapterResultError, match="PauliX"):
        canonical_to_pennylane_result(
            canonical_011(),
            [qml.expval(qml.Hadamard(0))],
            wire_order=(0, 1, 2),
            active_wires=(0, 1, 2),
        )
