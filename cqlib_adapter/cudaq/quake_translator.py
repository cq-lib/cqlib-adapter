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

"""Direct translation from CUDA-Q Quake MLIR to cqlib construction IR.

The translator intentionally supports the statically evaluable circuit subset
used by the adapter. Unsupported CUDA-Q language features fail explicitly;
this module never falls back to an OpenQASM representation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import cudaq
from cudaq.mlir.dialects import quake

from cqlib_adapter.common import AdapterConversionError

_MAX_STATIC_LOOP_ITERATIONS = 10_000
_SCALAR_ARITHMETIC: dict[str, Callable[[Any, Any], Any]] = {
    "arith.addi": lambda left, right: left + right,
    "arith.addf": lambda left, right: left + right,
    "arith.subi": lambda left, right: left - right,
    "arith.subf": lambda left, right: left - right,
    "arith.muli": lambda left, right: left * right,
    "arith.mulf": lambda left, right: left * right,
    "arith.divsi": lambda left, right: int(left / right),
    "arith.divui": lambda left, right: int(left / right),
    "arith.divf": lambda left, right: left / right,
    "arith.remsi": lambda left, right: left % right,
    "arith.remui": lambda left, right: left % right,
}
_UNARY_ARITHMETIC: dict[str, Callable[[Any], Any]] = {
    "arith.negf": lambda value: -value,
    "math.absf": abs,
    "math.cos": math.cos,
    "math.exp": math.exp,
    "math.sin": math.sin,
    "math.sqrt": math.sqrt,
}
_INTEGER_COMPARISONS: dict[int, Callable[[Any, Any], bool]] = {
    0: lambda left, right: left == right,
    1: lambda left, right: left != right,
    2: lambda left, right: left < right,
    3: lambda left, right: left <= right,
    4: lambda left, right: left > right,
    5: lambda left, right: left >= right,
    6: lambda left, right: int(left) < int(right),
    7: lambda left, right: int(left) <= int(right),
    8: lambda left, right: int(left) > int(right),
    9: lambda left, right: int(left) >= int(right),
}
_SELF_ADJOINT_GATES = frozenset({"h", "x", "y", "z", "swap"})


@dataclass(frozen=True, slots=True)
class QuakeTranslation:
    """The direct cqlib circuit plus facts recovered from Quake MLIR."""

    circuit: Any
    width: int
    measured_qubits: tuple[int, ...]
    measurement_bases: tuple[str, ...]
    auto_measure_all: bool
    allocation_widths: tuple[int, ...]
    entrypoint: str | None


@dataclass(frozen=True, slots=True)
class _Terminator:
    name: str
    values: tuple[Any, ...]


def _operation_name(operation: Any) -> str:
    return str(operation.operation.name)


def _attribute_value(attribute: Any) -> Any:
    value = getattr(attribute, "value", None)
    if value is not None:
        return value
    text = str(attribute)
    if text in {"true", "false"}:
        return text == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError as exc:
            raise AdapterConversionError(
                f"cannot statically evaluate MLIR attribute {text!r}"
            ) from exc


def _is_builder(kernel: Any) -> bool:
    py_kernel = getattr(cudaq, "PyKernel", None)
    if py_kernel is not None:
        try:
            return isinstance(kernel, py_kernel)
        except TypeError:
            pass
    return type(kernel).__name__ == "PyKernel"


def _concrete_module(kernel: Any, arguments: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
    if _is_builder(kernel):
        expected = int(getattr(kernel, "argument_count", len(getattr(kernel, "arguments", ()))))
        if len(arguments) != expected:
            raise AdapterConversionError(
                f"CUDA-Q builder expects {expected} arguments, received {len(arguments)}"
            )
        return kernel.module, arguments

    try:
        specialized = cudaq.synthesize(kernel, *arguments) if arguments else kernel
        if not getattr(specialized, "is_compiled", False):
            specialized.compile()
        module = specialized.qkeModule
    except Exception as exc:
        raise AdapterConversionError(
            f"failed to obtain concrete Quake MLIR from CUDA-Q kernel: {exc}"
        ) from exc
    return module, arguments


def _entrypoint(module: Any) -> Any:
    entries = [
        operation
        for operation in module.body.operations
        if _operation_name(operation) == "func.func" and "cudaq-entrypoint" in operation.attributes
    ]
    if len(entries) != 1:
        raise AdapterConversionError(
            f"expected exactly one CUDA-Q entrypoint in Quake MLIR, found {len(entries)}"
        )
    return entries[0]


def _entrypoint_name(operation: Any) -> str | None:
    attribute = operation.attributes.get("sym_name")
    return None if attribute is None else str(attribute).strip('"')


def _fixed_allocation_width(result_type: Any) -> int:
    if quake.RefType.isinstance(result_type):
        return 1
    if not quake.VeqType.isinstance(result_type):
        raise AdapterConversionError(
            f"unsupported quake.alloca result type {result_type}; expected !quake.ref "
            "or a fixed !quake.veq<N>"
        )
    if not quake.VeqType.hasSpecifiedSize(result_type):
        raise AdapterConversionError(
            "dynamic CUDA-Q qvector allocation is not supported; use a fixed width"
        )
    width = int(quake.VeqType.getSize(result_type))
    if width <= 0:
        raise AdapterConversionError("CUDA-Q kernel must allocate at least one qubit")
    return width


class _QuakeInterpreter:
    def __init__(self, block: Any, arguments: tuple[Any, ...]) -> None:
        self.block = block
        self.values: dict[Any, Any] = {}
        self.qubits: dict[Any, int | tuple[int, ...]] = {}
        self.measurements: list[int] = []
        self.measurement_bases: list[str] = []
        self.measurement_seen = False
        self.allocation_widths: list[int] = []

        block_arguments = tuple(block.arguments)
        if len(block_arguments) != len(arguments) and (block_arguments or arguments):
            raise AdapterConversionError(
                "CUDA-Q entrypoint scalar argument count does not match the supplied "
                f"arguments ({len(block_arguments)} != {len(arguments)})"
            )
        for value, argument in zip(block_arguments, arguments, strict=True):
            if isinstance(argument, bool | int | float):
                self.values[value] = argument
            else:
                raise AdapterConversionError(
                    "direct CUDA-Q conversion supports only bool, int and float "
                    f"entrypoint arguments, received {type(argument).__name__}"
                )

        next_qubit = 0
        for operation in block.operations:
            if _operation_name(operation) != "quake.alloca":
                continue
            if operation.operands:
                raise AdapterConversionError(
                    "dynamic quake.alloca operands are not supported; use fixed qalloc widths"
                )
            if len(operation.results) != 1:
                raise AdapterConversionError("quake.alloca must produce exactly one result")
            width = _fixed_allocation_width(operation.results[0].type)
            allocated = tuple(range(next_qubit, next_qubit + width))
            self.qubits[operation.results[0]] = allocated[0] if width == 1 else allocated
            self.allocation_widths.append(width)
            next_qubit += width

        if next_qubit <= 0:
            raise AdapterConversionError("CUDA-Q kernel must allocate at least one qubit")
        self.width = next_qubit
        self.circuit = import_module("cqlib").Circuit(next_qubit)

    def _value(self, value: Any) -> Any:
        try:
            return self.values[value]
        except KeyError:
            try:
                return self.qubits[value]
            except KeyError as exc:
                raise AdapterConversionError(
                    f"cannot statically resolve SSA value {value} in CUDA-Q kernel"
                ) from exc

    def _qubit_tuple(self, value: Any) -> tuple[int, ...]:
        resolved = self._value(value)
        if isinstance(resolved, int):
            return (resolved,)
        if isinstance(resolved, tuple) and all(isinstance(item, int) for item in resolved):
            return resolved
        raise AdapterConversionError(f"expected a quantum reference, received {resolved!r}")

    def _store_result(self, operation: Any, value: Any) -> None:
        if len(operation.results) != 1:
            raise AdapterConversionError(
                f"{_operation_name(operation)} must produce exactly one scalar result"
            )
        self.values[operation.results[0]] = value

    def _segments(self, operation: Any) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        attribute = operation.attributes.get("operandSegmentSizes")
        if attribute is None or len(attribute) != 3:
            raise AdapterConversionError(
                f"{_operation_name(operation)} has no valid params/control/target segmentation"
            )
        sizes = tuple(int(_attribute_value(attribute[index])) for index in range(3))
        if sum(sizes) != len(operation.operands):
            raise AdapterConversionError(
                f"{_operation_name(operation)} operand segmentation is inconsistent"
            )
        params_end = sizes[0]
        controls_end = params_end + sizes[1]
        operands = tuple(operation.operands)
        return (
            operands[:params_end],
            operands[params_end:controls_end],
            operands[controls_end:],
        )

    def _append_gate(self, operation: Any) -> None:
        operation_name = _operation_name(operation)
        gate = operation_name.removeprefix("quake.")
        params_values, control_values, target_values = self._segments(operation)
        params = tuple(float(self._value(value)) for value in params_values)
        controls = tuple(qubit for value in control_values for qubit in self._qubit_tuple(value))
        targets = tuple(qubit for value in target_values for qubit in self._qubit_tuple(value))
        is_adjoint = "is_adj" in operation.attributes

        if self.measurement_seen:
            raise AdapterConversionError(
                f"CUDA-Q accepts terminal measurements only; {operation_name} follows a measurement"
            )
        if is_adjoint:
            if gate in {"rx", "ry", "rz", "r1"} and len(params) == 1:
                params = (-params[0],)
            elif gate == "s":
                gate = "sdg"
            elif gate == "t":
                gate = "tdg"
            elif gate not in _SELF_ADJOINT_GATES:
                raise AdapterConversionError(
                    f"unsupported adjoint CUDA-Q operation {operation_name}"
                )

        if not controls:
            if gate in {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}:
                if len(targets) == 1 and not params:
                    getattr(self.circuit, gate)(targets[0])
                    return
            elif gate in {"rx", "ry", "rz"}:
                if len(targets) == 1 and len(params) == 1:
                    getattr(self.circuit, gate)(targets[0], params[0])
                    return
            elif gate == "r1":
                if len(targets) == 1 and len(params) == 1:
                    self.circuit.phase(targets[0], params[0])
                    return
            elif gate == "u3":
                if len(targets) == 1 and len(params) == 3:
                    self.circuit.u(targets[0], *params)
                    return
            elif gate == "swap" and len(targets) == 2 and not params:
                self.circuit.swap(*targets)
                return
        elif len(controls) == 1:
            if gate in {"x", "y", "z"} and len(targets) == 1 and not params:
                getattr(self.circuit, f"c{gate}")(controls[0], targets[0])
                return
            if gate in {"rx", "ry", "rz"} and len(targets) == 1 and len(params) == 1:
                getattr(self.circuit, f"c{gate}")(
                    controls[0],
                    targets[0],
                    params[0],
                )
                return
        elif gate == "x" and len(controls) == 2 and len(targets) == 1 and not params:
            self.circuit.ccx(controls[0], controls[1], targets[0])
            return

        raise AdapterConversionError(
            f"unsupported CUDA-Q operation {operation_name} with {len(params)} parameters, "
            f"{len(controls)} controls and {len(targets)} targets"
        )

    def _append_measurement(self, operation: Any, *, nested: bool) -> None:
        operation_name = _operation_name(operation)
        if nested:
            raise AdapterConversionError(
                f"{operation_name} inside CUDA-Q control flow is not supported"
            )
        basis = operation_name.removeprefix("quake.m").upper()
        qubits = tuple(
            qubit for operand in operation.operands for qubit in self._qubit_tuple(operand)
        )
        if not qubits:
            raise AdapterConversionError(f"{operation_name} has no measured qubits")
        for qubit in qubits:
            if qubit in self.measurements:
                raise AdapterConversionError(f"CUDA-Q qubit {qubit} is measured more than once")
            if basis not in {"X", "Y", "Z"}:
                raise AdapterConversionError(
                    f"unsupported CUDA-Q measurement operation {operation_name}"
                )
            self.measurements.append(qubit)
            self.measurement_bases.append(basis)
        self.measurement_seen = True

    def _comparison(self, operation: Any) -> None:
        predicate_attribute = operation.attributes.get("predicate")
        if predicate_attribute is None or len(operation.operands) != 2:
            raise AdapterConversionError("arith.cmpi must have a predicate and two operands")
        predicate = int(_attribute_value(predicate_attribute))
        comparison = _INTEGER_COMPARISONS.get(predicate)
        if comparison is None:
            raise AdapterConversionError(f"unsupported arith.cmpi predicate {predicate}")
        self._store_result(
            operation,
            comparison(
                self._value(operation.operands[0]),
                self._value(operation.operands[1]),
            ),
        )

    def _run_region(self, region: Any, incoming: tuple[Any, ...]) -> _Terminator:
        blocks = tuple(region.blocks)
        if len(blocks) != 1:
            raise AdapterConversionError(
                "direct CUDA-Q conversion supports single-block static control-flow regions"
            )
        block = blocks[0]
        if len(block.arguments) != len(incoming):
            raise AdapterConversionError("CUDA-Q control-flow block argument count is inconsistent")
        for argument, value in zip(block.arguments, incoming, strict=True):
            self.values[argument] = value
        terminator = self._run_operations(block.operations, nested=True)
        if terminator is None:
            raise AdapterConversionError("CUDA-Q control-flow region has no terminator")
        return terminator

    def _run_loop(self, operation: Any) -> None:
        if self.measurement_seen:
            raise AdapterConversionError("CUDA-Q cc.loop follows a measurement")
        regions = tuple(operation.regions)
        if len(regions) < 3:
            raise AdapterConversionError("CUDA-Q cc.loop must have condition, body and step")
        carried = tuple(self._value(operand) for operand in operation.operands)
        iteration = 0
        while True:
            condition = self._run_region(regions[0], carried)
            if condition.name != "cc.condition" or not condition.values:
                raise AdapterConversionError(
                    "CUDA-Q cc.loop condition region must end with cc.condition"
                )
            if not bool(condition.values[0]):
                carried = condition.values[1:] or carried
                break
            if iteration >= _MAX_STATIC_LOOP_ITERATIONS:
                raise AdapterConversionError(
                    "CUDA-Q static loop exceeds the 10000-iteration safety limit"
                )
            carried = condition.values[1:] or carried
            body = self._run_region(regions[1], carried)
            if body.name != "cc.continue":
                raise AdapterConversionError("CUDA-Q cc.loop body must end with cc.continue")
            carried = body.values or carried
            step = self._run_region(regions[2], carried)
            if step.name != "cc.continue":
                raise AdapterConversionError("CUDA-Q cc.loop step must end with cc.continue")
            carried = step.values or carried
            iteration += 1
        if len(operation.results) != len(carried) and (operation.results or carried):
            raise AdapterConversionError("CUDA-Q cc.loop result count is inconsistent")
        for result, value in zip(operation.results, carried, strict=True):
            self.values[result] = value

    def _run_operations(
        self,
        operations: Any,
        *,
        nested: bool,
    ) -> _Terminator | None:
        for operation in operations:
            name = _operation_name(operation)
            if name == "quake.alloca":
                if nested:
                    raise AdapterConversionError(
                        "quake.alloca inside CUDA-Q control flow is not supported"
                    )
                continue
            if name == "quake.extract_ref":
                if len(operation.operands) < 1 or len(operation.results) != 1:
                    raise AdapterConversionError("malformed quake.extract_ref operation")
                vector = self._qubit_tuple(operation.operands[0])
                raw_index = operation.attributes.get("rawIndex")
                if raw_index is not None and int(_attribute_value(raw_index)) >= 0:
                    index = int(_attribute_value(raw_index))
                elif len(operation.operands) == 2:
                    index = int(self._value(operation.operands[1]))
                else:
                    raise AdapterConversionError(
                        "quake.extract_ref index cannot be statically resolved"
                    )
                if index < 0 or index >= len(vector):
                    raise AdapterConversionError(
                        f"quake.extract_ref index {index} is outside qvector width {len(vector)}"
                    )
                self.qubits[operation.results[0]] = vector[index]
                continue
            if name == "arith.constant":
                attribute = operation.attributes.get("value")
                if attribute is None:
                    raise AdapterConversionError("arith.constant has no value attribute")
                self._store_result(operation, _attribute_value(attribute))
                continue
            if name in _SCALAR_ARITHMETIC:
                if len(operation.operands) != 2:
                    raise AdapterConversionError(f"{name} must have two operands")
                try:
                    result = _SCALAR_ARITHMETIC[name](
                        self._value(operation.operands[0]),
                        self._value(operation.operands[1]),
                    )
                except (ArithmeticError, TypeError, ValueError) as exc:
                    raise AdapterConversionError(
                        f"cannot statically evaluate {name}: {exc}"
                    ) from exc
                self._store_result(operation, result)
                continue
            if name in _UNARY_ARITHMETIC:
                if len(operation.operands) != 1:
                    raise AdapterConversionError(f"{name} must have one operand")
                self._store_result(
                    operation,
                    _UNARY_ARITHMETIC[name](self._value(operation.operands[0])),
                )
                continue
            if name == "arith.cmpi":
                self._comparison(operation)
                continue
            if name in {"arith.index_cast", "arith.sitofp", "arith.fptosi"}:
                if len(operation.operands) != 1:
                    raise AdapterConversionError(f"{name} must have one operand")
                value = self._value(operation.operands[0])
                converted = float(value) if name == "arith.sitofp" else int(value)
                self._store_result(operation, converted)
                continue
            if name == "cc.loop":
                self._run_loop(operation)
                continue
            if name in {"quake.mx", "quake.my", "quake.mz"}:
                self._append_measurement(operation, nested=nested)
                continue
            if name == "quake.reset":
                if self.measurement_seen:
                    raise AdapterConversionError("quake.reset follows a measurement")
                qubits = tuple(
                    qubit for operand in operation.operands for qubit in self._qubit_tuple(operand)
                )
                if len(qubits) != 1:
                    raise AdapterConversionError("quake.reset must target one qubit")
                self.circuit.reset(qubits[0])
                continue
            # CUDA-Q >= 0.16 injects quake.log_output output markers; they carry
            # no circuit semantics and are safe to ignore.
            if name in {"func.return", "quake.dealloc", "quake.log_output"}:
                continue
            if name.startswith("quake."):
                self._append_gate(operation)
                continue
            if name in {"cc.condition", "cc.continue"}:
                return _Terminator(
                    name,
                    tuple(self._value(operand) for operand in operation.operands),
                )
            raise AdapterConversionError(
                f"unsupported CUDA-Q Quake operation {name}; no operation was skipped"
            )
        return None

    def translate(self) -> QuakeTranslation:
        self._run_operations(self.block.operations, nested=False)
        added_measurements = not self.measurements
        if added_measurements:
            for qubit in range(self.width):
                self.measurements.append(qubit)
                self.measurement_bases.append("Z")
        for qubit, basis in zip(
            self.measurements,
            self.measurement_bases,
            strict=True,
        ):
            if basis == "X":
                self.circuit.h(qubit)
            elif basis == "Y":
                self.circuit.sdg(qubit)
                self.circuit.h(qubit)
        for qubit in self.measurements:
            self.circuit.measure(qubit)
        self.circuit.validate()
        return QuakeTranslation(
            circuit=self.circuit,
            width=self.width,
            measured_qubits=tuple(self.measurements),
            measurement_bases=tuple(self.measurement_bases),
            auto_measure_all=added_measurements,
            allocation_widths=tuple(self.allocation_widths),
            entrypoint=None,
        )


def translate_quake_to_cqlib(kernel: Any, *arguments: Any) -> QuakeTranslation:
    """Translate a concrete CUDA-Q kernel directly from Quake MLIR to cqlib."""

    module, entry_arguments = _concrete_module(kernel, tuple(arguments))
    entrypoint = _entrypoint(module)
    blocks = tuple(entrypoint.regions[0].blocks)
    if len(blocks) != 1:
        raise AdapterConversionError("CUDA-Q entrypoint must contain exactly one block")
    interpreter = _QuakeInterpreter(blocks[0], entry_arguments)
    result = interpreter.translate()
    return QuakeTranslation(
        circuit=result.circuit,
        width=result.width,
        measured_qubits=result.measured_qubits,
        measurement_bases=result.measurement_bases,
        auto_measure_all=result.auto_measure_all,
        allocation_widths=result.allocation_widths,
        entrypoint=_entrypoint_name(entrypoint),
    )


__all__ = ["QuakeTranslation", "translate_quake_to_cqlib"]
