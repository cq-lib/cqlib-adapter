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

"""Translate CUDA-Q kernels directly from Quake MLIR into cqlib IR."""

from __future__ import annotations

from typing import Any, cast

import cudaq

from cqlib_adapter.common import (
    AdapterConversionError,
    CircuitCompiler,
    CompilationArtifact,
    CompilationOptions,
    MeasurementMetadata,
    MeasurementSlot,
    NormalizedDevice,
    TranslationBundle,
    TranslationMetadata,
)
from cqlib_adapter.common.typing import CircuitLike

from .quake_translator import translate_quake_to_cqlib


def _is_parameterized_builder(kernel: Any) -> bool:
    """Recognize the CUDA-Q 0.15 ``make_kernel``/``PyKernel`` surface."""

    py_kernel = getattr(cudaq, "PyKernel", None)
    if py_kernel is not None:
        try:
            if isinstance(kernel, py_kernel):
                return bool(getattr(kernel, "argument_count", 0))
        except TypeError:
            pass
    return type(kernel).__name__ == "PyKernel" and bool(
        getattr(kernel, "argument_count", len(getattr(kernel, "arguments", ())))
    )


def _kernel_name(kernel: Any) -> str | None:
    name = getattr(kernel, "name", None) or getattr(kernel, "__name__", None)
    return str(name) if name else None


def cudaq_to_openqasm(kernel: Any, *arguments: Any) -> str:
    """Export OpenQASM 2 for diagnostics, independently of adapter conversion.

    Production conversion uses Quake MLIR directly and never calls or falls
    back to this exporter. CUDA-Q 0.15 cannot specialize a parameterized
    ``PyKernel`` builder for OpenQASM 2; decorated kernels are specialized with
    :func:`cudaq.synthesize`.
    """

    if _is_parameterized_builder(kernel):
        raise AdapterConversionError(
            "parameterized CUDA-Q PyKernel builders cannot be specialized to OpenQASM 2 "
            "with CUDA-Q 0.15; use cudaq_to_cqlib for direct Quake conversion"
        )
    try:
        specialized = cudaq.synthesize(kernel, *arguments) if arguments else kernel
        qasm = str(cudaq.translate(specialized, format="openqasm2")).strip()
    except AdapterConversionError:
        raise
    except Exception as exc:
        raise AdapterConversionError(
            "failed to export CUDA-Q kernel as OpenQASM 2; the diagnostic exporter "
            "requires a concrete kernel and supports parameters only through a typed "
            f"@cudaq.kernel: {exc}"
        ) from exc
    if not qasm:
        raise AdapterConversionError("CUDA-Q produced empty OpenQASM 2 output")
    return qasm


def cudaq_to_cqlib(kernel: Any, *arguments: Any) -> TranslationBundle[CircuitLike]:
    """Translate a statically evaluable CUDA-Q kernel directly into cqlib IR."""

    try:
        translated = translate_quake_to_cqlib(kernel, *arguments)
    except AdapterConversionError:
        raise
    except Exception as exc:
        raise AdapterConversionError(
            f"failed to translate CUDA-Q Quake MLIR directly to cqlib: {exc}"
        ) from exc

    qubit_ids = tuple(f"q{index}" for index in range(translated.width))
    slots = tuple(
        MeasurementSlot(qubit_ids[qubit], classical_bit, "__global__")
        for classical_bit, qubit in enumerate(translated.measured_qubits)
    )
    metadata = TranslationMetadata(
        framework="cudaq",
        qubits=qubit_ids,
        measurements=MeasurementMetadata(slots, len(slots), {"__global__": len(slots)}),
        circuit_name=_kernel_name(kernel),
        warnings=(
            ("kernel had no explicit measurement; the adapter added final mz for every qubit",)
            if translated.auto_measure_all
            else ()
        ),
        extras={
            "source_ir": "quake",
            "quake_entrypoint": translated.entrypoint,
            "auto_measure_all": translated.auto_measure_all,
            "allocation_widths": translated.allocation_widths,
            "measured_qubits": translated.measured_qubits,
            "measurement_bases": translated.measurement_bases,
        },
    )
    return TranslationBundle(cast(CircuitLike, translated.circuit), metadata)


def compile_cudaq_kernel(
    kernel: Any,
    *arguments: Any,
    device: NormalizedDevice | None = None,
    options: CompilationOptions | None = None,
    compiler: CircuitCompiler | None = None,
    circuit_index: int | None = None,
) -> CompilationArtifact:
    """Translate and compile one CUDA-Q kernel into validated QCIS."""

    return (compiler or CircuitCompiler()).compile(
        cudaq_to_cqlib(kernel, *arguments),
        device=device,
        options=options,
        circuit_index=circuit_index,
    )


__all__ = ["compile_cudaq_kernel", "cudaq_to_cqlib", "cudaq_to_openqasm"]
