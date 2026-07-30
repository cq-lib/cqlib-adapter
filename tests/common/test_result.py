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

from cqlib_adapter.common import (
    AdapterJobFailedError,
    AdapterResultError,
    CompilationArtifact,
    CompiledMeasurement,
    MeasurementMetadata,
    MeasurementSlot,
    ResultConverter,
    TranslationMetadata,
)

from .fakes import FakeCircuit, FakeExecutionResult, FakeStatus

pytestmark = pytest.mark.unit


def _artifact(
    measurements: tuple[CompiledMeasurement, ...] = (
        CompiledMeasurement(2, 1),
        CompiledMeasurement(0, 0),
    ),
    *,
    num_classical_bits: int = 2,
) -> CompilationArtifact:
    slots = tuple(
        MeasurementSlot(f"q{index}", measurement.classical_bit)
        for index, measurement in enumerate(measurements)
    )
    metadata = TranslationMetadata(
        "qiskit",
        tuple(slot.logical_qubit for slot in slots),
        MeasurementMetadata(slots, num_classical_bits),
    )
    return CompilationArtifact(
        qcis="M Q2\nM Q0",
        circuit=FakeCircuit((("measure_bit", (2,)), ("measure_bit", (0,)))),
        metadata=metadata,
        device=None,
        changed=False,
        steps=(),
        measurements=measurements,
    )


def test_result_restores_classical_order_from_cqlib_little_endian() -> None:
    raw = FakeExecutionResult(
        task_id="task",
        shots=100,
        qubit_indices=(2, 0),
        counts={"10": 60, "01": 40},
    )
    converted = ResultConverter().convert(raw, _artifact())
    assert converted.counts == {"01": 60, "10": 40}
    assert converted.probabilities == {"01": 0.6, "10": 0.4}
    assert converted.samples[:2] == ((1, 0), (1, 0))
    assert len(converted.samples) == 100


def test_result_normalizes_cloud_probabilities() -> None:
    raw = FakeExecutionResult(
        task_id="task",
        shots=2,
        qubit_indices=(2, 0),
        counts={"10": 1, "01": 1},
        probabilities={"10": 2.0, "01": 1.0},
    )
    converted = ResultConverter().convert(raw, _artifact())
    assert converted.probabilities["01"] == pytest.approx(2 / 3)
    assert converted.probabilities["10"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"10": 50, "01": 49}, {"01": 51, "10": 49}),
        ({"10": 51, "01": 50}, {"01": 50, "10": 50}),
    ],
)
def test_result_reconciles_small_probability_rounding_difference(
    counts: dict[str, int],
    expected: dict[str, int],
) -> None:
    raw = FakeExecutionResult(
        task_id="calibrated-task",
        shots=100,
        qubit_indices=(2, 0),
        counts=counts,
        probabilities={outcome: count / sum(counts.values()) for outcome, count in counts.items()},
    )

    converted = ResultConverter().convert(raw, _artifact())

    assert converted.counts == expected
    assert sum(converted.counts.values()) == 100
    assert len(converted.samples) == 100


def test_result_rejects_difference_larger_than_probability_rounding_bound() -> None:
    raw = FakeExecutionResult(
        task_id="broken-task",
        shots=100,
        qubit_indices=(2, 0),
        counts={"10": 45, "01": 45},
        probabilities={"10": 0.5, "01": 0.5},
    )

    with pytest.raises(AdapterResultError, match="counts total 90 does not equal shots 100"):
        ResultConverter().convert(raw, _artifact())


def test_result_does_not_reconcile_mismatch_without_probabilities() -> None:
    raw = FakeExecutionResult(
        task_id="raw-task",
        shots=100,
        qubit_indices=(2, 0),
        counts={"10": 50, "01": 49},
    )

    with pytest.raises(AdapterResultError, match="counts total 99 does not equal shots 100"):
        ResultConverter().convert(raw, _artifact())


def test_unmeasured_classical_bits_default_to_zero() -> None:
    artifact = _artifact(
        (CompiledMeasurement(2, 1), CompiledMeasurement(0, 0)),
        num_classical_bits=3,
    )
    raw = FakeExecutionResult("task", 1, (2, 0), {"10": 1})
    assert ResultConverter().convert(raw, artifact).counts == {"001": 1}


def test_irrelevant_physical_bits_are_aggregated() -> None:
    artifact = _artifact((CompiledMeasurement(2, 0),), num_classical_bits=1)
    raw = FakeExecutionResult("task", 5, (2, 1), {"00": 2, "10": 3})
    assert ResultConverter().convert(raw, artifact).counts == {"0": 5}


def test_failed_cloud_status_preserves_error_details() -> None:
    raw = FakeExecutionResult(
        "failed-task",
        1,
        (2, 0),
        {},
        status=FakeStatus("failed", "hardware fault", 503),
    )
    with pytest.raises(AdapterJobFailedError) as raised:
        ResultConverter().convert(raw, _artifact())
    assert "hardware fault [code=503]" in str(raised.value)
    assert "task_id=failed-task" in str(raised.value)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (FakeExecutionResult("t", 1, (2, 0), {"1": 1}), "expected 2"),
        (FakeExecutionResult("t", 1, (2, 0), {"2x": 1}), "binary digits"),
        (FakeExecutionResult("t", 2, (2, 0), {"00": 1}), "does not equal shots"),
        (FakeExecutionResult("t", 1, (2, 0), {"00": -1}), "invalid count"),
        (FakeExecutionResult("t", 1, (2, 0), {"00": True}), "invalid count"),
    ],
)
def test_malformed_counts_are_rejected(
    raw: FakeExecutionResult,
    message: str,
) -> None:
    with pytest.raises(AdapterResultError, match=message):
        ResultConverter().convert(raw, _artifact())


def test_missing_measured_physical_qubit_is_rejected() -> None:
    raw = FakeExecutionResult("t", 1, (0, 1), {"00": 1})
    with pytest.raises(AdapterResultError, match="omits measured physical qubit 2"):
        ResultConverter().convert(raw, _artifact())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_invalid_probabilities_are_rejected(value: float) -> None:
    raw = FakeExecutionResult(
        "t",
        1,
        (2, 0),
        {"00": 1},
        probabilities={"00": value},
    )
    with pytest.raises(AdapterResultError, match="invalid probability"):
        ResultConverter().convert(raw, _artifact())


def test_artifact_without_final_measurements_is_rejected() -> None:
    metadata = TranslationMetadata("cirq", ())
    artifact = CompilationArtifact(
        "X Q0",
        FakeCircuit((("X", (0,)),)),
        metadata,
        None,
        False,
        (),
        (),
    )
    raw = FakeExecutionResult("t", 1, (0,), {"0": 1})
    with pytest.raises(AdapterResultError, match="no final measurement"):
        ResultConverter().convert(raw, artifact)


def test_canonical_result_mappings_are_immutable() -> None:
    raw = FakeExecutionResult("t", 1, (2, 0), {"00": 1})
    result = ResultConverter().convert(raw, _artifact())
    with pytest.raises(TypeError):
        result.counts["11"] = 2  # type: ignore[index]
