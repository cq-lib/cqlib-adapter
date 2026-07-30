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

"""Framework-independent compilation, cloud, job, and result core."""

from .circuit import (
    CompiledMeasurement,
    MeasurementMetadata,
    MeasurementSlot,
    QubitMapping,
    TranslationBundle,
    TranslationMetadata,
)
from .compiler import CircuitCompiler, CompilationArtifact, DefaultCqlibRuntime
from .device import Coupling, DeviceStatus, DeviceToll, NormalizedDevice, qubit_index
from .errors import (
    AdapterCompileError,
    AdapterConversionError,
    AdapterDeviceError,
    AdapterError,
    AdapterJobError,
    AdapterJobFailedError,
    AdapterJobTimeoutError,
    AdapterResultError,
    AdapterSubmissionError,
    ErrorContext,
)
from .job import AdapterJob, JobSnapshot, JobState
from .options import (
    CalibrationMode,
    CompilationMode,
    CompilationOptions,
    RunOptions,
    require_positive_finite,
)
from .platform import TianyanConnector
from .result import CanonicalResult, ResultConverter

__all__ = [
    "AdapterCompileError",
    "AdapterConversionError",
    "AdapterDeviceError",
    "AdapterError",
    "AdapterJob",
    "AdapterJobError",
    "AdapterJobFailedError",
    "AdapterJobTimeoutError",
    "AdapterResultError",
    "AdapterSubmissionError",
    "CalibrationMode",
    "CanonicalResult",
    "CircuitCompiler",
    "CompilationArtifact",
    "CompilationMode",
    "CompilationOptions",
    "CompiledMeasurement",
    "Coupling",
    "DefaultCqlibRuntime",
    "DeviceStatus",
    "DeviceToll",
    "ErrorContext",
    "JobSnapshot",
    "JobState",
    "MeasurementMetadata",
    "MeasurementSlot",
    "NormalizedDevice",
    "QubitMapping",
    "ResultConverter",
    "RunOptions",
    "TianyanConnector",
    "TranslationBundle",
    "TranslationMetadata",
    "qubit_index",
    "require_positive_finite",
]
