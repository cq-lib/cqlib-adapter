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

"""Shared exception hierarchy for every framework adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Optional identifiers attached to an adapter failure."""

    device_name: str | None = None
    task_id: str | None = None
    circuit_index: int | None = None

    def render(self) -> str:
        fields: list[str] = []
        if self.device_name is not None:
            fields.append(f"device={self.device_name}")
        if self.task_id is not None:
            fields.append(f"task_id={self.task_id}")
        if self.circuit_index is not None:
            fields.append(f"circuit_index={self.circuit_index}")
        return ", ".join(fields)


class AdapterError(Exception):
    """Base error raised at the framework-independent adapter boundary."""

    stage: ClassVar[str] = "adapter"

    def __init__(self, message: str, *, context: ErrorContext | None = None) -> None:
        self.message = message
        self.context = context or ErrorContext()
        details = self.context.render()
        rendered = f"{self.stage}: {message}"
        super().__init__(f"{rendered} ({details})" if details else rendered)


class AdapterConversionError(AdapterError):
    """A framework circuit cannot be translated."""

    stage = "conversion"


class AdapterCompileError(AdapterError):
    """cqlib compilation or QCIS validation failed."""

    stage = "compilation"


class AdapterDeviceError(AdapterError):
    """The requested cloud device is invalid or unavailable."""

    stage = "device"


class AdapterSubmissionError(AdapterError):
    """The request could not be submitted to Tianyan."""

    stage = "submission"


class AdapterJobError(AdapterError):
    """Base class for cloud-job failures."""

    stage = "job"


class AdapterJobTimeoutError(AdapterJobError, TimeoutError):
    """Waiting for a Tianyan task exceeded the configured timeout."""


class AdapterJobFailedError(AdapterJobError):
    """A Tianyan task reached an unsuccessful terminal state."""


class AdapterResultError(AdapterError):
    """A cloud result is incomplete, inconsistent, or malformed."""

    stage = "result"


__all__ = [
    "AdapterCompileError",
    "AdapterConversionError",
    "AdapterDeviceError",
    "AdapterError",
    "AdapterJobError",
    "AdapterJobFailedError",
    "AdapterJobTimeoutError",
    "AdapterResultError",
    "AdapterSubmissionError",
    "ErrorContext",
]
