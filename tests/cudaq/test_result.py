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

from cqlib_adapter.common import AdapterResultError, CanonicalResult
from cqlib_adapter.cudaq.result import CudaQSampleResult, canonical_to_cudaq_result

pytestmark = pytest.mark.cudaq


def canonical(counts: dict[str, int], *, shots: int, width: int) -> CanonicalResult:
    return CanonicalResult(
        task_id="cudaq-result",
        shots=shots,
        counts=counts,
        probabilities={outcome: count / shots for outcome, count in counts.items()},
        num_classical_bits=width,
    )


def test_canonical_011_is_displayed_as_cudaq_110() -> None:
    result = canonical_to_cudaq_result(canonical({"011": 10}, shots=10, width=3))

    assert isinstance(result, CudaQSampleResult)
    assert dict(result) == {"110": 10}
    assert result.count("110") == 10
    assert result.count("000") == 0
    assert result.probability("110") == 1.0
    assert result.shots_count == 10
    assert result.get_total_shots() == 10
    assert result.num_qubits == 3
    assert result.register_names == ["__global__"]
    assert result.get_register_counts() is result


def test_sample_result_mapping_and_expectation_z_match_cudaq_behavior() -> None:
    result = CudaQSampleResult({"00": 3, "01": 1, "11": 4})

    assert list(result) == ["00", "01", "11"]
    assert list(result.values()) == [3, 1, 4]
    assert result.expectation_z() == pytest.approx(0.75)
    assert result.expectation() == pytest.approx(0.75)
    assert result.most_probable() == "11"
    assert "CudaQSampleResult" in repr(result)
    with pytest.raises(KeyError, match="unknown CUDA-Q measurement register"):
        result.count("00", "separate")


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({}, "must not be empty"),
        ({"0": 1, "00": 1}, "one positive width"),
        ({"02": 1}, "invalid CUDA-Q sample outcome"),
        ({"00": -1}, "invalid CUDA-Q sample count"),
        ({"00": 0}, "at least one shot"),
    ],
)
def test_invalid_sample_counts_are_rejected(counts: dict[str, int], message: str) -> None:
    with pytest.raises(AdapterResultError, match=message):
        CudaQSampleResult(counts)


def test_invalid_canonical_width_and_total_are_rejected() -> None:
    with pytest.raises(AdapterResultError, match="measured bits"):
        canonical_to_cudaq_result(canonical({"": 1}, shots=1, width=0))
    with pytest.raises(AdapterResultError, match="outcome width"):
        canonical_to_cudaq_result(canonical({"0": 1}, shots=1, width=2))
    with pytest.raises(AdapterResultError, match="counts total"):
        canonical_to_cudaq_result(canonical({"0": 1}, shots=2, width=1))
