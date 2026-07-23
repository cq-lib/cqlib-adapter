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

"""Convert cqlib execution data into one stable, framework-neutral result."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .compiler import CompilationArtifact
from .device import qubit_index
from .errors import (
    AdapterJobFailedError,
    AdapterResultError,
    ErrorContext,
)
from .typing import ExecutionResultLike


@dataclass(frozen=True, slots=True)
class CanonicalResult:
    """MSB-left classical counts plus samples in ascending classical-bit order."""

    task_id: str
    shots: int
    counts: Mapping[str, int]
    probabilities: Mapping[str, float]
    num_classical_bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(
            self,
            "probabilities",
            MappingProxyType(dict(self.probabilities)),
        )

    @property
    def samples(self) -> tuple[tuple[int, ...], ...]:
        """Expand counts deterministically; columns are c0, c1, ..."""

        rows: list[tuple[int, ...]] = []
        for bitstring, count in sorted(self.counts.items()):
            row = tuple(int(bit) for bit in reversed(bitstring))
            rows.extend([row] * count)
        return tuple(rows)


def _status_text(result: ExecutionResultLike) -> str:
    return str(result.status.kind).strip().lower().rsplit(".", 1)[-1]


def _physical_bits(
    outcome: str,
    result: ExecutionResultLike,
) -> dict[int, str]:
    bits = outcome.replace(" ", "")
    if len(bits) != len(result.qubits) or set(bits) - {"0", "1"}:
        raise AdapterResultError(
            f"invalid cloud outcome {outcome!r}; expected {len(result.qubits)} binary digits"
        )
    return {qubit_index(qubit): bits[-1 - position] for position, qubit in enumerate(result.qubits)}


def _canonical_outcome(
    outcome: str,
    result: ExecutionResultLike,
    artifact: CompilationArtifact,
) -> str:
    physical = _physical_bits(outcome, result)
    classical = ["0"] * artifact.num_classical_bits
    for measurement in artifact.measurements:
        try:
            classical[measurement.classical_bit] = physical[measurement.physical_qubit]
        except KeyError as exc:
            raise AdapterResultError(
                f"cloud result omits measured physical qubit {measurement.physical_qubit}"
            ) from exc
    return "".join(reversed(classical))


def _reconcile_rounded_counts(
    counts: Mapping[str, int],
    shots: int,
    *,
    has_probabilities: bool,
    source_outcome_count: int,
    context: ErrorContext,
) -> dict[str, int]:
    """Restore totals changed only by cqlib-tianyan probability rounding."""

    observed = sum(counts.values())
    if observed == shots:
        return dict(counts)
    # cqlib-tianyan converts calibrated probabilities back to integer counts
    # with round(p * shots). Across N retained outcomes, the total rounding
    # discrepancy cannot exceed floor(N / 2). Keep larger inconsistencies fatal.
    maximum_rounding_error = source_outcome_count // 2
    if not has_probabilities or observed <= 0 or abs(observed - shots) > maximum_rounding_error:
        raise AdapterResultError(
            f"counts total {observed} does not equal shots {shots}",
            context=context,
        )

    quotas = {outcome: count * shots / observed for outcome, count in counts.items()}
    reconciled = {outcome: math.floor(quota) for outcome, quota in quotas.items()}
    remaining = shots - sum(reconciled.values())
    order = sorted(
        quotas,
        key=lambda outcome: (-(quotas[outcome] - reconciled[outcome]), outcome),
    )
    for outcome in order[:remaining]:
        reconciled[outcome] += 1
    return reconciled


class ResultConverter:
    """Validate one Tianyan result and restore framework classical ordering."""

    _SUCCESS = frozenset({"completed", "success", "succeeded", "done", "finished"})

    def convert(
        self,
        result: ExecutionResultLike,
        artifact: CompilationArtifact,
    ) -> CanonicalResult:
        context = ErrorContext(
            device_name=artifact.device.name if artifact.device is not None else None,
            task_id=result.task_id,
        )
        status = _status_text(result)
        if status not in self._SUCCESS:
            message = result.status.error_msg or f"task has non-success status {status!r}"
            code = result.status.error_code
            if code is not None:
                message = f"{message} [code={code}]"
            raise AdapterJobFailedError(message, context=context)
        if result.shots <= 0:
            raise AdapterResultError("cloud result shots must be positive", context=context)
        if not artifact.measurements:
            raise AdapterResultError(
                "compiled artifact has no final measurement mapping",
                context=context,
            )

        counts: dict[str, int] = {}
        for outcome, count in result.counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise AdapterResultError(
                    f"invalid count {count!r} for outcome {outcome!r}",
                    context=context,
                )
            canonical = _canonical_outcome(outcome, result, artifact)
            counts[canonical] = counts.get(canonical, 0) + count
        counts = _reconcile_rounded_counts(
            counts,
            result.shots,
            has_probabilities=bool(result.probabilities),
            source_outcome_count=len(result.counts),
            context=context,
        )

        probabilities = self._probabilities(result, artifact, counts, context)
        return CanonicalResult(
            task_id=result.task_id,
            shots=result.shots,
            counts=counts,
            probabilities=probabilities,
            num_classical_bits=artifact.num_classical_bits,
        )

    def _probabilities(
        self,
        result: ExecutionResultLike,
        artifact: CompilationArtifact,
        counts: Mapping[str, int],
        context: ErrorContext,
    ) -> dict[str, float]:
        if not result.probabilities:
            return {outcome: count / result.shots for outcome, count in counts.items()}

        converted: dict[str, float] = {}
        for outcome, probability in result.probabilities.items():
            value = float(probability)
            if not math.isfinite(value) or value < 0:
                raise AdapterResultError(
                    f"invalid probability {probability!r} for outcome {outcome!r}",
                    context=context,
                )
            canonical = _canonical_outcome(outcome, result, artifact)
            converted[canonical] = converted.get(canonical, 0.0) + value
        total = sum(converted.values())
        if not math.isfinite(total) or total <= 0:
            raise AdapterResultError(
                "cloud probabilities must have a positive finite sum",
                context=context,
            )
        return {outcome: probability / total for outcome, probability in converted.items()}


__all__ = ["CanonicalResult", "ResultConverter"]
