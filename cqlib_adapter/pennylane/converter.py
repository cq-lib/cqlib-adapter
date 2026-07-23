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

"""Conversion between PennyLane QuantumScripts and cqlib construction IR."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import pennylane as qml
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

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

from .operations import RXY, X2M, X2P, XY, XY2M, XY2P, Y2M, Y2P, FSim

_SINGLE_NO_PARAM = {
    "Identity": "i",
    "Hadamard": "h",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "S": "s",
    "Adjoint(S)": "sdg",
    "T": "t",
    "Adjoint(T)": "tdg",
    "SX": "x2p",
    "Adjoint(SX)": "x2m",
    "X2P": "x2p",
    "X2M": "x2m",
    "Y2P": "y2p",
    "Y2M": "y2m",
}
_SINGLE_ONE_PARAM = {
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "PhaseShift": "phase",
    "XY": "xy",
    "XY2P": "xy2p",
    "XY2M": "xy2m",
}
_TWO_NO_PARAM = {
    "CNOT": "cx",
    "CY": "cy",
    "CZ": "cz",
    "SWAP": "swap",
}
_TWO_ONE_PARAM = {
    "IsingXX": "rxx",
    "IsingYY": "ryy",
    "IsingZZ": "rzz",
    "IsingZX": "rzx",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
}
_SUPPORTED_MEASUREMENTS = (
    CountsMP,
    ProbabilityMP,
    SampleMP,
    ExpectationMP,
    VarianceMP,
)
_OBSERVABLE_BASES = {
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
}
SUPPORTED_OPERATION_NAMES = frozenset(
    set(_SINGLE_NO_PARAM)
    | set(_SINGLE_ONE_PARAM)
    | set(_TWO_NO_PARAM)
    | set(_TWO_ONE_PARAM)
    | {"RXY", "FSim", "Toffoli", "Barrier", "GlobalPhase"}
)


def _numeric(value: Any, *, operation: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterConversionError(
            f"PennyLane operation {operation!r} has a non-scalar parameter {value!r}"
        ) from exc


def _validated_wire_order(tape: QuantumScript, wire_order: Any | None) -> Wires:
    order = Wires(tape.wires if wire_order is None else wire_order)
    if not order:
        raise AdapterConversionError("PennyLane circuit must contain at least one wire")
    if len(set(order)) != len(order):
        raise AdapterConversionError("PennyLane wire_order must contain unique labels")
    missing = [wire for wire in tape.wires if wire not in order]
    if missing:
        raise AdapterConversionError(f"wire_order omits PennyLane wires {missing!r}")
    return order


def _measurement_basis_requirements(
    tape: QuantumScript,
    order: Wires,
) -> dict[Any, str]:
    if not tape.measurements:
        raise AdapterConversionError("PennyLane circuit must contain a final measurement")
    requirements: dict[Any, str] = {}
    for measurement in tape.measurements:
        if not isinstance(measurement, _SUPPORTED_MEASUREMENTS):
            raise AdapterConversionError(
                f"unsupported PennyLane measurement {type(measurement).__name__!r}; "
                "supported measurements are counts, probs, sample, expval and var"
            )
        observable = measurement.obs
        required: tuple[tuple[Any, str], ...]
        if observable is None:
            if isinstance(measurement, ExpectationMP | VarianceMP):
                raise AdapterConversionError(
                    f"{type(measurement).__name__} requires a Pauli observable"
                )
            requested = tuple(measurement.wires) or tuple(tape.wires) or tuple(order)
            required = tuple((wire, "Z") for wire in requested)
        else:
            basis = _OBSERVABLE_BASES.get(str(observable.name))
            if basis is None or len(observable.wires) != 1:
                raise AdapterConversionError(
                    "observable-valued PennyLane measurements support only single-wire "
                    "PauliX, PauliY and PauliZ observables"
                )
            required = ((observable.wires[0], basis),)

        for wire, basis in required:
            previous = requirements.get(wire)
            if previous is not None and previous != basis:
                raise AdapterConversionError(
                    f"wire {wire!r} is requested in incompatible {previous} and {basis} "
                    "measurement bases; split the measurements into separate circuits"
                )
            requirements[wire] = basis
    return requirements


def supports_measurement(measurement: Any) -> bool:
    """Return whether one measurement has a directly supported result shape."""

    if not isinstance(measurement, _SUPPORTED_MEASUREMENTS):
        return False
    if measurement.obs is None:
        return not isinstance(measurement, ExpectationMP | VarianceMP)
    return str(measurement.obs.name) in _OBSERVABLE_BASES and len(measurement.obs.wires) == 1


def _append_measurement_basis_rotation(output: Any, qubit: int, basis: str) -> None:
    if basis == "X":
        output.h(qubit)
    elif basis == "Y":
        output.sdg(qubit)
        output.h(qubit)
    elif basis != "Z":
        raise AdapterConversionError(f"unsupported measurement basis {basis!r}")


def _append_cqlib_operation(
    output: Any,
    *,
    name: str,
    qubits: tuple[int, ...],
    params: tuple[float, ...],
) -> None:
    if name in _SINGLE_NO_PARAM and len(qubits) == 1 and not params:
        getattr(output, _SINGLE_NO_PARAM[name])(qubits[0])
        return
    if name in _SINGLE_ONE_PARAM and len(qubits) == 1 and len(params) == 1:
        getattr(output, _SINGLE_ONE_PARAM[name])(qubits[0], params[0])
        return
    if name == "RXY" and len(qubits) == 1 and len(params) == 2:
        output.rxy(qubits[0], params[0], params[1])
        return
    if name in _TWO_NO_PARAM and len(qubits) == 2 and not params:
        getattr(output, _TWO_NO_PARAM[name])(qubits[0], qubits[1])
        return
    if name in _TWO_ONE_PARAM and len(qubits) == 2 and len(params) == 1:
        getattr(output, _TWO_ONE_PARAM[name])(qubits[0], qubits[1], params[0])
        return
    if name == "FSim" and len(qubits) == 2 and len(params) == 2:
        output.fsim(qubits[0], qubits[1], params[0], params[1])
        return
    if name == "Toffoli" and len(qubits) == 3 and not params:
        output.ccx(qubits[0], qubits[1], qubits[2])
        return
    raise AdapterConversionError(
        f"unsupported PennyLane operation {name!r} with "
        f"{len(qubits)} wires and {len(params)} parameters"
    )


def pennylane_to_cqlib(
    tape: QuantumScript,
    *,
    wire_order: Any | None = None,
) -> TranslationBundle[CircuitLike]:
    """Translate a finite-shot PennyLane QuantumScript to cqlib IR.

    All wires in ``wire_order`` are measured once at the end. Framework
    measurement processes are retained as metadata and evaluated from the
    canonical shot data after execution.
    """

    if not isinstance(tape, QuantumScript):
        raise TypeError("tape must be a pennylane.tape.QuantumScript")
    order = _validated_wire_order(tape, wire_order)
    basis_requirements = _measurement_basis_requirements(tape, order)
    wire_map = {wire: index for index, wire in enumerate(order)}
    qubit_ids = tuple(f"wire{index}" for index in range(len(order)))
    circuit_type = import_module("cqlib").Circuit
    output = circuit_type(len(order))

    try:
        for index, operation in enumerate(tape.operations):
            name = operation.name
            qubits = tuple(wire_map[wire] for wire in operation.wires)
            try:
                params = tuple(_numeric(value, operation=name) for value in operation.data)
            except AdapterConversionError as exc:
                raise AdapterConversionError(f"operation at index {index}: {exc.message}") from exc
            if name == "Barrier":
                output.barrier(list(qubits))
                continue
            if name == "GlobalPhase":
                if qubits or len(params) != 1:
                    raise AdapterConversionError("GlobalPhase must have one scalar parameter")
                output.set_global_phase(params[0])
                continue
            try:
                _append_cqlib_operation(output, name=name, qubits=qubits, params=params)
            except AdapterConversionError as exc:
                raise AdapterConversionError(f"operation at index {index}: {exc.message}") from exc

        for wire, basis in basis_requirements.items():
            _append_measurement_basis_rotation(output, wire_map[wire], basis)

        slots = tuple(
            MeasurementSlot(logical_qubit, classical_bit, "pennylane")
            for classical_bit, logical_qubit in enumerate(qubit_ids)
        )
        for qubit in range(len(order)):
            output.measure(qubit)
        output.validate()
        metadata = TranslationMetadata(
            framework="pennylane",
            qubits=qubit_ids,
            measurements=MeasurementMetadata(slots, len(order)),
            circuit_name=getattr(tape, "name", None),
            parameter_names=(),
            extras={
                "wire_order": tuple(order),
                "active_wire_order": tuple(tape.wires),
                "measurement_types": tuple(type(item).__name__ for item in tape.measurements),
                "measurement_bases": tuple(basis_requirements.get(wire, "Z") for wire in order),
            },
        )
        return TranslationBundle(cast(CircuitLike, output), metadata)
    except AdapterConversionError:
        raise
    except Exception as exc:
        raise AdapterConversionError(f"failed to translate PennyLane circuit: {exc}") from exc


def compile_pennylane_circuit(
    tape: QuantumScript,
    *,
    wire_order: Any | None = None,
    device: NormalizedDevice | None = None,
    options: CompilationOptions | None = None,
    compiler: CircuitCompiler | None = None,
    circuit_index: int | None = None,
) -> CompilationArtifact:
    """Translate and compile one PennyLane QuantumScript into validated QCIS."""

    return (compiler or CircuitCompiler()).compile(
        pennylane_to_cqlib(tape, wire_order=wire_order),
        device=device,
        options=options,
        circuit_index=circuit_index,
    )


def _operation_parts(operation: Any) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    value_instruction = operation.instruction
    instruction = getattr(value_instruction, "instruction", value_instruction)
    if instruction is None:
        raise AdapterConversionError("cqlib classical control is not supported")
    name = str(instruction.name).strip().lower()
    params = tuple(
        float(parameter.evaluate()) if hasattr(parameter, "evaluate") else float(parameter)
        for parameter in operation.params
    )
    return name, tuple(qubit.index for qubit in operation.qubits), params


def _pennylane_operation(name: str, qubits: tuple[Any, ...], params: tuple[float, ...]) -> Any:
    no_param = {
        "i": qml.Identity,
        "h": qml.Hadamard,
        "x": qml.PauliX,
        "y": qml.PauliY,
        "z": qml.PauliZ,
        "s": qml.S,
        "t": qml.T,
        "x2p": X2P,
        "x2m": X2M,
        "y2p": Y2P,
        "y2m": Y2M,
    }
    one_param = {
        "rx": qml.RX,
        "ry": qml.RY,
        "rz": qml.RZ,
        "phase": qml.PhaseShift,
        "xy": XY,
        "xy2p": XY2P,
        "xy2m": XY2M,
    }
    two_no_param = {"cx": qml.CNOT, "cy": qml.CY, "cz": qml.CZ, "swap": qml.SWAP}
    two_one_param = {
        "rxx": qml.IsingXX,
        "ryy": qml.IsingYY,
        "rzz": qml.IsingZZ,
        "crx": qml.CRX,
        "cry": qml.CRY,
        "crz": qml.CRZ,
    }
    if name in no_param and len(qubits) == 1 and not params:
        return no_param[name](wires=qubits)
    if name == "sdg" and len(qubits) == 1:
        return qml.adjoint(qml.S)(wires=qubits)
    if name == "tdg" and len(qubits) == 1:
        return qml.adjoint(qml.T)(wires=qubits)
    if name in one_param and len(qubits) == 1 and len(params) == 1:
        return one_param[name](params[0], wires=qubits)
    if name == "rxy" and len(qubits) == 1 and len(params) == 2:
        return RXY(*params, wires=qubits)
    if name in two_no_param and len(qubits) == 2 and not params:
        return two_no_param[name](wires=qubits)
    if name == "rzx" and len(qubits) == 2 and len(params) == 1:
        return qml.PauliRot(params[0], "ZX", wires=qubits)
    if name in two_one_param and len(qubits) == 2 and len(params) == 1:
        return two_one_param[name](params[0], wires=qubits)
    if name == "fsim" and len(qubits) == 2 and len(params) == 2:
        return FSim(*params, wires=qubits)
    if name == "ccx" and len(qubits) == 3 and not params:
        return qml.Toffoli(wires=qubits)
    raise AdapterConversionError(
        f"unsupported cqlib instruction {name!r} during PennyLane conversion"
    )


def cqlib_to_pennylane(
    circuit: CircuitLike,
    *,
    metadata: TranslationMetadata | None = None,
    shots: int | None = None,
    measurement: str = "counts",
) -> QuantumScript:
    """Convert supported cqlib construction IR to a PennyLane QuantumScript."""

    circuit.validate()
    physical = tuple(qubit.index for qubit in circuit.qubits)
    default_order = tuple(range(max(physical, default=-1) + 1))
    if metadata is not None and "wire_order" in metadata.extras:
        wire_order = tuple(metadata.extras["wire_order"])
    else:
        wire_order = default_order
    operations: list[Any] = []
    measured: list[Any] = []
    for operation in circuit.operations:
        name, indices, params = _operation_parts(operation)
        wires = tuple(wire_order[index] if index < len(wire_order) else index for index in indices)
        if name in {"measure", "measure_bit", "measure_bits"}:
            measured.extend(wire for wire in wires if wire not in measured)
        elif name in {"barrier", "b"}:
            operations.append(qml.Barrier(wires=wires, only_visual=True))
        elif name == "reset":
            raise AdapterConversionError("cqlib reset conversion is not supported")
        else:
            operations.append(_pennylane_operation(name, wires, params))
    result_wires = measured or list(wire_order)
    measurement_name = measurement.strip().lower()
    if measurement_name == "counts":
        measurements = [qml.counts(wires=result_wires)]
    elif measurement_name == "probs":
        measurements = [qml.probs(wires=result_wires)]
    elif measurement_name == "sample":
        measurements = [qml.sample(wires=result_wires)]
    else:
        raise ValueError("measurement must be 'counts', 'probs' or 'sample'")
    return QuantumScript(operations, measurements, shots=shots)


__all__ = [
    "SUPPORTED_OPERATION_NAMES",
    "compile_pennylane_circuit",
    "cqlib_to_pennylane",
    "pennylane_to_cqlib",
    "supports_measurement",
]
