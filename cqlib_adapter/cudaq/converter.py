"""Translate CUDA-Q kernels through OpenQASM 2 into cqlib IR."""

from __future__ import annotations

import re
from importlib import import_module
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

_QREG = re.compile(r"\bqreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*]\s*;", re.I)
_CREG = re.compile(r"\bcreg\s+[A-Za-z_][A-Za-z0-9_]*\s*\[\s*\d+\s*]\s*;", re.I)
_MEASURE = re.compile(
    r"\bmeasure\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*\[\s*(\d+)\s*])?"
    r"\s*->\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s*\[\s*(\d+)\s*])?\s*;",
    re.I,
)
_CLASSICAL_CONTROL = re.compile(r"\bif\s*\(", re.I)


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
    """Specialize a CUDA-Q kernel and export its OpenQASM 2 program.

    CUDA-Q 0.15 cannot specialize a parameterized ``PyKernel`` builder for
    OpenQASM 2. Decorated ``@cudaq.kernel`` functions are specialized first
    with :func:`cudaq.synthesize`, which is the supported M5 parameter path.
    """

    if _is_parameterized_builder(kernel):
        raise AdapterConversionError(
            "parameterized CUDA-Q PyKernel builders cannot be specialized to OpenQASM 2 "
            "with CUDA-Q 0.15; use a typed @cudaq.kernel and pass its arguments"
        )
    try:
        specialized = cudaq.synthesize(kernel, *arguments) if arguments else kernel
        qasm = str(cudaq.translate(specialized, format="openqasm2")).strip()
    except AdapterConversionError:
        raise
    except Exception as exc:
        detail = str(exc)
        raise AdapterConversionError(
            "failed to export CUDA-Q kernel as OpenQASM 2; M5 requires a concrete kernel "
            "with one qalloc/qvector and supports parameters only through a typed "
            f"@cudaq.kernel: {detail}"
        ) from exc
    if not qasm:
        raise AdapterConversionError("CUDA-Q produced empty OpenQASM 2 output")
    return qasm


def _single_qreg(qasm: str) -> tuple[str, int]:
    registers = [(match.group(1), int(match.group(2))) for match in _QREG.finditer(qasm)]
    if len(registers) != 1:
        raise AdapterConversionError(
            "CUDA-Q M5 requires exactly one qalloc/qvector; "
            f"OpenQASM 2 declared {len(registers)} quantum registers"
        )
    name, width = registers[0]
    if width <= 0:
        raise AdapterConversionError("CUDA-Q kernel must allocate at least one qubit")
    return name, width


def _measurement_qubits(qasm: str, qreg_name: str, width: int) -> tuple[int, ...]:
    measured: list[int] = []
    for match in _MEASURE.finditer(qasm):
        source_name, source_index, _destination_name, destination_index = match.groups()
        if source_name != qreg_name:
            raise AdapterConversionError(
                f"CUDA-Q measurement references unknown quantum register {source_name!r}"
            )
        if source_index is None:
            if destination_index is not None:
                raise AdapterConversionError(
                    "OpenQASM register measurement must target a full classical register"
                )
            measured.extend(range(width))
        else:
            index = int(source_index)
            if index >= width:
                raise AdapterConversionError(
                    f"CUDA-Q measurement qubit index {index} exceeds kernel width {width}"
                )
            measured.append(index)
    if len(set(measured)) != len(measured):
        raise AdapterConversionError(
            "CUDA-Q M5 does not support measuring one qubit more than once"
        )
    return tuple(measured)


def _ensure_final_measurements(
    qasm: str,
    qreg_name: str,
    width: int,
) -> tuple[str, tuple[int, ...], bool]:
    if _CLASSICAL_CONTROL.search(qasm):
        raise AdapterConversionError(
            "CUDA-Q M5 does not support OpenQASM classical control or mid-circuit feedback"
        )
    measured = _measurement_qubits(qasm, qreg_name, width)
    if measured:
        last_measurement = max(match.end() for match in _MEASURE.finditer(qasm))
        trailing = re.sub(r"//[^\n]*|/\*.*?\*/", "", qasm[last_measurement:], flags=re.S)
        statements = [item.strip() for item in trailing.split(";") if item.strip()]
        if statements and any(not item.lower().startswith("barrier") for item in statements):
            raise AdapterConversionError(
                "CUDA-Q M5 accepts terminal measurements only; an operation follows mz"
            )
        return qasm, measured, False

    separator = "" if qasm.endswith("\n") else "\n"
    measured_qasm = f"{qasm}{separator}creg cqlib_mz[{width}];\nmeasure {qreg_name} -> cqlib_mz;"
    return measured_qasm, tuple(range(width)), True


def cudaq_to_cqlib(kernel: Any, *arguments: Any) -> TranslationBundle[CircuitLike]:
    """Translate one concrete CUDA-Q kernel into validated cqlib construction IR."""

    original_qasm = cudaq_to_openqasm(kernel, *arguments)
    qreg_name, width = _single_qreg(original_qasm)
    qasm, measured, added_measurements = _ensure_final_measurements(original_qasm, qreg_name, width)
    qubit_ids = tuple(f"q{index}" for index in range(width))
    slots = tuple(
        MeasurementSlot(qubit_ids[qubit], classical_bit, "__global__")
        for classical_bit, qubit in enumerate(measured)
    )
    try:
        # cqlib's QASM parser models classical registers with ``store``
        # operations, while QCIS intentionally has no classical storage
        # instruction. Parse the quantum portion and recreate terminal
        # measurements through cqlib's public Circuit API.
        unitary_qasm = _CREG.sub("", _MEASURE.sub("", qasm))
        circuit = import_module("cqlib.ir.qasm2").loads(unitary_qasm)
        for qubit in measured:
            circuit.measure(qubit)
        circuit.validate()
    except Exception as exc:
        raise AdapterConversionError(
            f"cqlib could not parse CUDA-Q OpenQASM 2 output: {exc}"
        ) from exc
    metadata = TranslationMetadata(
        framework="cudaq",
        qubits=qubit_ids,
        measurements=MeasurementMetadata(slots, len(slots), {"__global__": len(slots)}),
        circuit_name=_kernel_name(kernel),
        warnings=(
            ("kernel had no explicit mz; the adapter added final measurement of every qubit",)
            if added_measurements
            else ()
        ),
        extras={
            "openqasm2": qasm,
            "source_openqasm2": original_qasm,
            "unitary_openqasm2": unitary_qasm,
            "auto_measure_all": added_measurements,
            "qreg_name": qreg_name,
            "measured_qubits": measured,
        },
    )
    return TranslationBundle(cast(CircuitLike, circuit), metadata)


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
