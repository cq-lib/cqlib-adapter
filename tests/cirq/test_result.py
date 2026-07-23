from __future__ import annotations

import cirq
import numpy as np
import pytest

from cqlib_adapter.cirq import canonical_to_cirq_probabilities, canonical_to_cirq_result
from cqlib_adapter.common import (
    AdapterResultError,
    CanonicalResult,
    MeasurementMetadata,
    MeasurementSlot,
    TranslationMetadata,
)

pytestmark = pytest.mark.cirq


def metadata() -> TranslationMetadata:
    return TranslationMetadata(
        framework="cirq",
        qubits=("q0", "q1", "q2"),
        measurements=MeasurementMetadata(
            (
                MeasurementSlot("q0", 0, "left"),
                MeasurementSlot("q1", 1, "pair"),
                MeasurementSlot("q2", 2, "pair"),
            ),
            3,
            {"left": 1, "pair": 2},
        ),
        extras={"measurement_key_bits": (("left", (0,)), ("pair", (1, 2)))},
    )


def test_result_dict_uses_cirq_measurement_key_column_order() -> None:
    canonical = CanonicalResult("task", 4, {"110": 4}, {"110": 1.0}, 3)

    result = canonical_to_cirq_result(canonical, metadata(), params={"theta": 0.5})

    assert isinstance(result, cirq.ResultDict)
    assert result.params == cirq.ParamResolver({"theta": 0.5})
    np.testing.assert_array_equal(result.measurements["left"], [[False]] * 4)
    np.testing.assert_array_equal(result.measurements["pair"], [[True, True]] * 4)
    assert result.histogram(key="left") == {0: 4}
    assert result.histogram(key="pair") == {3: 4}


def test_result_dict_preserves_shot_rows_for_multiple_outcomes() -> None:
    canonical = CanonicalResult(
        "task",
        3,
        {"000": 1, "101": 2},
        {"000": 1 / 3, "101": 2 / 3},
        3,
    )

    result = canonical_to_cirq_result(canonical, metadata())

    assert result.repetitions == 3
    assert result.measurements["left"].shape == (3, 1)
    assert result.measurements["pair"].shape == (3, 2)
    assert result.multi_measurement_histogram(keys=["left", "pair"]) == {
        (0, 0): 1,
        (1, 1): 2,
    }


def test_probabilities_are_marginalized_in_measurement_key_order() -> None:
    canonical = CanonicalResult(
        "task",
        4,
        {"000": 1, "101": 3},
        {"000": 0.25, "101": 0.75},
        3,
    )

    assert canonical_to_cirq_probabilities(
        canonical,
        metadata(),
        key="left",
    ) == {0: 0.25, 1: 0.75}
    assert canonical_to_cirq_probabilities(
        canonical,
        metadata(),
        key="pair",
    ) == {0: 0.25, 1: 0.75}
    with pytest.raises(ValueError, match="key is required"):
        canonical_to_cirq_probabilities(canonical, metadata())
    with pytest.raises(KeyError, match="unknown Cirq measurement key"):
        canonical_to_cirq_probabilities(canonical, metadata(), key="missing")


def test_result_width_and_measurement_metadata_errors_are_rejected() -> None:
    with pytest.raises(AdapterResultError, match="width"):
        canonical_to_cirq_result(
            CanonicalResult("task", 1, {"0": 1}, {"0": 1.0}, 1),
            metadata(),
        )
    broken = TranslationMetadata(
        framework="cirq",
        qubits=("q0",),
        measurements=MeasurementMetadata(
            (MeasurementSlot("q0", 0, "m"),),
            1,
            {"m": 1},
        ),
        extras={"measurement_key_bits": (("m", (2,)),)},
    )
    with pytest.raises(AdapterResultError, match="invalid classical bit"):
        canonical_to_cirq_result(
            CanonicalResult("task", 1, {"0": 1}, {"0": 1.0}, 1),
            broken,
        )
