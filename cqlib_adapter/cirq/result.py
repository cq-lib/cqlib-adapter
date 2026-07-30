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

"""Convert canonical Tianyan data into Cirq ResultDict values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cirq
import numpy as np

from cqlib_adapter.common import AdapterResultError, CanonicalResult, TranslationMetadata


def _measurement_key_bits(metadata: TranslationMetadata) -> dict[str, tuple[int, ...]]:
    stored = metadata.extras.get("measurement_key_bits")
    if stored is not None:
        result = {str(key): tuple(int(bit) for bit in bits) for key, bits in stored}
    else:
        grouped: dict[str, list[int]] = {}
        for slot in metadata.measurements.slots:
            grouped.setdefault(slot.key or "m", []).append(slot.classical_bit)
        result = {key: tuple(bits) for key, bits in grouped.items()}
    if not result:
        raise AdapterResultError("Cirq result metadata contains no measurement keys")
    flattened = tuple(bit for bits in result.values() for bit in bits)
    if len(set(flattened)) != len(flattened):
        raise AdapterResultError("Cirq measurement keys contain duplicate classical bits")
    if any(bit < 0 or bit >= metadata.measurements.num_classical_bits for bit in flattened):
        raise AdapterResultError("Cirq measurement key references an invalid classical bit")
    return result


def canonical_to_cirq_result(
    canonical: CanonicalResult,
    metadata: TranslationMetadata,
    *,
    params: cirq.ParamResolver | Mapping[str, Any] | None = None,
) -> cirq.ResultDict:
    """Return one Cirq ResultDict with columns in each measurement gate's qubit order."""

    if canonical.num_classical_bits != metadata.measurements.num_classical_bits:
        raise AdapterResultError("canonical result width does not match Cirq measurement metadata")
    rows = canonical.samples
    if len(rows) != canonical.shots:
        raise AdapterResultError("canonical sample count does not match shots")
    key_bits = _measurement_key_bits(metadata)
    measurements: dict[str, np.ndarray] = {}
    for key, bits in key_bits.items():
        measurements[key] = np.asarray(
            [[bool(row[index]) for index in bits] for row in rows],
            dtype=np.bool_,
        ).reshape(canonical.shots, len(bits))
    return cirq.ResultDict(params=cirq.ParamResolver(params), measurements=measurements)


def canonical_to_cirq_probabilities(
    canonical: CanonicalResult,
    metadata: TranslationMetadata,
    *,
    key: str | None = None,
) -> dict[int, float]:
    """Return a big-endian probability distribution for one measurement key."""

    if canonical.num_classical_bits != metadata.measurements.num_classical_bits:
        raise AdapterResultError("canonical result width does not match Cirq measurement metadata")
    key_bits = _measurement_key_bits(metadata)
    if key is None:
        if len(key_bits) != 1:
            raise ValueError("key is required when a Cirq circuit has multiple measurement keys")
        key = next(iter(key_bits))
    if key not in key_bits:
        raise KeyError(f"unknown Cirq measurement key {key!r}")
    probabilities: dict[int, float] = {}
    for outcome, probability in canonical.probabilities.items():
        selected = "".join(outcome[-1 - index] for index in key_bits[key])
        value = int(selected, 2) if selected else 0
        probabilities[value] = probabilities.get(value, 0.0) + float(probability)
    return dict(sorted(probabilities.items()))


__all__ = ["canonical_to_cirq_probabilities", "canonical_to_cirq_result"]
