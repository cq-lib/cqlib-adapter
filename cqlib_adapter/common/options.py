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

"""Validated options shared by all framework surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CompilationMode(StrEnum):
    """Compilation depth exposed by cqlib."""

    NORMAL = "normal"
    ENHANCED = "enhanced"


class CalibrationMode(StrEnum):
    """Whether Tianyan calibration or error mitigation is requested."""

    AUTO = "auto"
    ENABLED = "enabled"
    DISABLED = "disabled"


def require_positive_finite(value: object, *, name: str) -> float:
    """Return a finite positive numeric value or raise a stable input error."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return normalized


@dataclass(frozen=True, slots=True)
class CompilationOptions:
    """Framework-independent arguments passed to cqlib compilation."""

    mode: CompilationMode = CompilationMode.NORMAL
    target_basis: tuple[str, ...] | None = None
    initial_layout: Any | None = None
    resource_policy: Any | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        if self.target_basis is None:
            return
        normalized = tuple(gate.strip().upper() for gate in self.target_basis)
        if not normalized or any(not gate for gate in normalized):
            raise ValueError("target_basis must contain at least one non-empty gate name")
        if len(set(normalized)) != len(normalized):
            raise ValueError("target_basis must not contain duplicate gate names")
        object.__setattr__(self, "target_basis", normalized)


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Cloud execution options common to the four adapters."""

    device_name: str
    shots: int = 1024
    timeout: float = 120.0
    poll_interval: float = 5.0
    calibration: CalibrationMode = CalibrationMode.AUTO
    require_available: bool = True

    def __post_init__(self) -> None:
        if not self.device_name.strip():
            raise ValueError("device_name must not be empty")
        object.__setattr__(self, "device_name", self.device_name.strip())
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0:
            raise ValueError("shots must be a positive integer")
        object.__setattr__(self, "timeout", require_positive_finite(self.timeout, name="timeout"))
        object.__setattr__(
            self,
            "poll_interval",
            require_positive_finite(self.poll_interval, name="poll_interval"),
        )


__all__ = [
    "CalibrationMode",
    "CompilationMode",
    "CompilationOptions",
    "RunOptions",
    "require_positive_finite",
]
