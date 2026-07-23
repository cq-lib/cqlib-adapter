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

"""CUDA-Q-compatible sampling results built from canonical Tianyan data."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

from cqlib_adapter.common import AdapterResultError, CanonicalResult


class CudaQSampleResult(Mapping[str, int]):
    """Public, serialization-independent subset of ``cudaq.SampleResult``.

    CUDA-Q 0.15 does not expose a supported counts constructor. This object
    intentionally implements the familiar mapping, count and probability
    behavior without using CUDA-Q's private serialized representation.
    """

    def __init__(self, counts: Mapping[str, int]) -> None:
        converted = dict(counts)
        if not converted:
            raise AdapterResultError("CUDA-Q sample counts must not be empty")
        widths = {len(outcome) for outcome in converted}
        if len(widths) != 1 or next(iter(widths)) <= 0:
            raise AdapterResultError("CUDA-Q sample outcomes must have one positive width")
        for outcome, count in converted.items():
            if set(outcome) - {"0", "1"}:
                raise AdapterResultError(f"invalid CUDA-Q sample outcome {outcome!r}")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise AdapterResultError(
                    f"invalid CUDA-Q sample count {count!r} for outcome {outcome!r}"
                )
        shots = sum(converted.values())
        if shots <= 0:
            raise AdapterResultError("CUDA-Q sample counts must contain at least one shot")
        self._counts = MappingProxyType(dict(sorted(converted.items())))
        self._shots = shots
        self._num_qubits = next(iter(widths))

    def __getitem__(self, bitstring: str) -> int:
        return self._counts[bitstring]

    def __iter__(self) -> Iterator[str]:
        return iter(self._counts)

    def __len__(self) -> int:
        return len(self._counts)

    @property
    def shots_count(self) -> int:
        return self._shots

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def get_total_shots(self) -> int:
        return self._shots

    @property
    def register_names(self) -> list[str]:
        return ["__global__"]

    def count(self, bitstring: str, register_name: str = "__global__") -> int:
        self._check_register(register_name)
        return self._counts.get(bitstring, 0)

    def probability(self, bitstring: str, register_name: str = "__global__") -> float:
        self._check_register(register_name)
        return self.count(bitstring) / self._shots

    def expectation_z(self, register_name: str = "__global__") -> float:
        self._check_register(register_name)
        return sum(
            ((-1.0) ** outcome.count("1")) * count / self._shots
            for outcome, count in self._counts.items()
        )

    def expectation(self, register_name: str = "__global__") -> float:
        return self.expectation_z(register_name)

    def most_probable(self, register_name: str = "__global__") -> str:
        self._check_register(register_name)
        return min(self._counts, key=lambda outcome: (-self._counts[outcome], outcome))

    def get_register_counts(self, register_name: str = "__global__") -> CudaQSampleResult:
        self._check_register(register_name)
        return self

    def dump(self, register_name: str = "__global__") -> None:
        self._check_register(register_name)
        print(dict(self._counts))

    @staticmethod
    def _check_register(register_name: str) -> None:
        if register_name != "__global__":
            raise KeyError(f"unknown CUDA-Q measurement register {register_name!r}")

    def __repr__(self) -> str:
        return f"CudaQSampleResult({dict(self._counts)!r})"


def canonical_to_cudaq_result(canonical: CanonicalResult) -> CudaQSampleResult:
    """Convert canonical MSB-left storage to CUDA-Q's q0-left display order."""

    if canonical.num_classical_bits <= 0:
        raise AdapterResultError("canonical CUDA-Q result must contain measured bits")
    counts: dict[str, int] = {}
    for outcome, count in canonical.counts.items():
        if len(outcome) != canonical.num_classical_bits:
            raise AdapterResultError(
                "canonical CUDA-Q outcome width does not match num_classical_bits"
            )
        cudaq_outcome = outcome[::-1]
        counts[cudaq_outcome] = counts.get(cudaq_outcome, 0) + count
    if sum(counts.values()) != canonical.shots:
        raise AdapterResultError("canonical CUDA-Q counts total does not match shots")
    return CudaQSampleResult(counts)


__all__ = ["CudaQSampleResult", "canonical_to_cudaq_result"]
