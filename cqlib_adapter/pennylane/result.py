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

"""Convert canonical Tianyan data to PennyLane measurement return values."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)

from cqlib_adapter.common import AdapterResultError, CanonicalResult


def _row(bitstring: str) -> tuple[int, ...]:
    return tuple(int(bit) for bit in reversed(bitstring))


def _wire_indices(
    measurement: Any,
    wire_map: dict[Any, int],
    active_wires: Sequence[Any],
) -> tuple[int, ...]:
    requested = tuple(measurement.wires) or tuple(active_wires)
    if not requested:
        raise AdapterResultError("PennyLane measurement resolves to no wires")
    try:
        return tuple(wire_map[wire] for wire in requested)
    except KeyError as exc:
        raise AdapterResultError(
            f"PennyLane result references unknown wire {exc.args[0]!r}"
        ) from exc


def _bits(row: tuple[int, ...], indices: tuple[int, ...]) -> str:
    return "".join(str(row[index]) for index in indices)


def _counts(
    canonical: CanonicalResult,
    indices: tuple[int, ...],
    *,
    all_outcomes: bool,
) -> dict[str, int]:
    converted: dict[str, int] = {}
    if all_outcomes:
        converted.update({"".join(bits): 0 for bits in product("01", repeat=len(indices))})
    for outcome, count in canonical.counts.items():
        key = _bits(_row(outcome), indices)
        converted[key] = converted.get(key, 0) + count
    return converted


def _probabilities(canonical: CanonicalResult, indices: tuple[int, ...]) -> np.ndarray:
    values = np.zeros(1 << len(indices), dtype=float)
    for outcome, probability in canonical.probabilities.items():
        selected = _bits(_row(outcome), indices)
        values[int(selected, 2)] += probability
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise AdapterResultError("PennyLane probabilities must have a positive finite sum")
    return values / total


def _samples(canonical: CanonicalResult, indices: tuple[int, ...]) -> np.ndarray:
    rows = canonical.samples
    return np.asarray([[row[index] for index in indices] for row in rows], dtype=np.int8)


def _observable_eigenvalues(measurement: Any) -> np.ndarray | None:
    observable = measurement.obs
    if observable is None:
        return None
    if str(observable.name) not in {"PauliX", "PauliY", "PauliZ"} or len(observable.wires) != 1:
        raise AdapterResultError(
            "observable-valued PennyLane results support only single-wire "
            "PauliX, PauliY and PauliZ observables"
        )
    values = np.asarray(measurement.eigvals(), dtype=float).reshape(-1)
    if values.shape != (2,):
        raise AdapterResultError("Pauli observable must expose exactly two eigenvalues")
    return values


def _observable_counts(
    canonical: CanonicalResult,
    indices: tuple[int, ...],
    eigenvalues: np.ndarray,
    *,
    all_outcomes: bool,
) -> dict[float, int]:
    bit_counts = _counts(canonical, indices, all_outcomes=False)
    converted: dict[float, int] = {}
    if all_outcomes:
        converted.update({float(value): 0 for value in eigenvalues})
    for bitstring, count in bit_counts.items():
        value = float(eigenvalues[int(bitstring, 2)])
        converted[value] = converted.get(value, 0) + count
    return converted


def canonical_to_pennylane_result(
    canonical: CanonicalResult,
    measurements: Sequence[Any],
    *,
    wire_order: Sequence[Any],
    active_wires: Sequence[Any],
) -> Any:
    """Return counts, probs and/or sample in PennyLane wire order.

    Canonical rows are ordered by logical classical bits ``c0, c1, ...``.
    PennyLane bitstrings instead list the requested wires from left to right;
    this explicit projection is what prevents accidental bit reversal.
    """

    order = tuple(wire_order)
    if canonical.num_classical_bits != len(order):
        raise AdapterResultError(
            "canonical result width does not match the PennyLane device wire order"
        )
    if not measurements:
        raise AdapterResultError("PennyLane circuit has no measurement process")
    wire_map = {wire: index for index, wire in enumerate(order)}
    results: list[Any] = []
    for measurement in measurements:
        indices = _wire_indices(measurement, wire_map, active_wires)
        eigenvalues = _observable_eigenvalues(measurement)
        if isinstance(measurement, CountsMP):
            if eigenvalues is None:
                results.append(
                    _counts(
                        canonical,
                        indices,
                        all_outcomes=bool(measurement.all_outcomes),
                    )
                )
            else:
                results.append(
                    _observable_counts(
                        canonical,
                        indices,
                        eigenvalues,
                        all_outcomes=bool(measurement.all_outcomes),
                    )
                )
        elif isinstance(measurement, ProbabilityMP):
            results.append(_probabilities(canonical, indices))
        elif isinstance(measurement, SampleMP):
            samples = _samples(canonical, indices)
            if eigenvalues is None:
                results.append(samples)
            else:
                results.append(eigenvalues[samples[:, 0]])
        elif isinstance(measurement, ExpectationMP | VarianceMP):
            if eigenvalues is None:
                raise AdapterResultError(
                    f"{type(measurement).__name__} requires a supported observable"
                )
            probabilities = _probabilities(canonical, indices)
            mean = float(np.dot(probabilities, eigenvalues))
            if isinstance(measurement, ExpectationMP):
                results.append(mean)
            else:
                second_moment = float(np.dot(probabilities, eigenvalues**2))
                results.append(second_moment - mean**2)
        else:
            raise AdapterResultError(
                f"unsupported PennyLane measurement {type(measurement).__name__!r}"
            )
    return results[0] if len(results) == 1 else tuple(results)


__all__ = ["canonical_to_pennylane_result"]
