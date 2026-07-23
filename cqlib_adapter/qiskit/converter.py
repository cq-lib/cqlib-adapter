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

"""Conversion between Qiskit circuits and the cqlib construction IR."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister

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

from .gates import FSimGate, RXYGate, XY2MGate, XY2PGate, XYGate

_SINGLE_NO_PARAM = {
    "id": "i",
    "i": "i",
    "h": "h",
    "x": "x",
    "y": "y",
    "z": "z",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "x2p": "x2p",
    "x2m": "x2m",
    "y2p": "y2p",
    "y2m": "y2m",
}
_SINGLE_ONE_PARAM = {
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "p": "phase",
    "phase": "phase",
    "xy": "xy",
    "xy2p": "xy2p",
    "xy2m": "xy2m",
}
_TWO_NO_PARAM = {
    "cx": "cx",
    "cy": "cy",
    "cz": "cz",
    "swap": "swap",
}
_TWO_ONE_PARAM = {
    "rxx": "rxx",
    "ryy": "ryy",
    "rzz": "rzz",
    "rzx": "rzx",
    "crx": "crx",
    "cry": "cry",
    "crz": "crz",
}


def _numeric(value: Any, *, instruction: str) -> float:
    parameters = getattr(value, "parameters", ())
    if parameters:
        names = sorted(str(parameter) for parameter in parameters)
        raise AdapterConversionError(
            f"instruction {instruction!r} has unbound parameters {names}; "
            "bind the Qiskit circuit before conversion"
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterConversionError(
            f"instruction {instruction!r} has a non-numeric parameter {value!r}"
        ) from exc


def _qubit_ids(circuit: QuantumCircuit) -> tuple[str, ...]:
    return tuple(f"q{index}" for index in range(circuit.num_qubits))


def _measurement_key(circuit: QuantumCircuit, clbit: Any) -> str | None:
    location = circuit.find_bit(clbit)
    return location.registers[0][0].name if location.registers else None


def _append_cqlib_gate(
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
    if name == "rxy" and len(qubits) == 1 and len(params) == 2:
        output.rxy(qubits[0], params[0], params[1])
        return
    if name == "u" and len(qubits) == 1 and len(params) == 3:
        output.u(qubits[0], params[0], params[1], params[2])
        return
    if name in _TWO_NO_PARAM and len(qubits) == 2 and not params:
        getattr(output, _TWO_NO_PARAM[name])(qubits[0], qubits[1])
        return
    if name in _TWO_ONE_PARAM and len(qubits) == 2 and len(params) == 1:
        getattr(output, _TWO_ONE_PARAM[name])(qubits[0], qubits[1], params[0])
        return
    if name == "fsim" and len(qubits) == 2 and len(params) == 2:
        output.fsim(qubits[0], qubits[1], params[0], params[1])
        return
    if name == "ccx" and len(qubits) == 3 and not params:
        output.ccx(qubits[0], qubits[1], qubits[2])
        return
    raise AdapterConversionError(
        f"unsupported Qiskit instruction {name!r} with "
        f"{len(qubits)} qubits and {len(params)} parameters"
    )


def qiskit_to_cqlib(circuit: QuantumCircuit) -> TranslationBundle[CircuitLike]:
    """Translate a fully bound Qiskit circuit to real cqlib construction IR.

    The adapter supports unitary gates, barriers, reset and final measurements.
    Dynamic control flow, conditions, initialize and gates after a measurement
    are rejected with an actionable conversion error.
    """

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a qiskit.QuantumCircuit")
    if circuit.num_qubits <= 0:
        raise AdapterConversionError("Qiskit circuit must contain at least one qubit")
    if circuit.parameters:
        raise AdapterConversionError(
            "Qiskit circuit contains unbound parameters; call assign_parameters first"
        )

    circuit_type = import_module("cqlib").Circuit
    output = circuit_type(circuit.num_qubits)
    qubit_ids = _qubit_ids(circuit)
    slots: list[MeasurementSlot] = []
    measurement_seen = False

    try:
        phase = _numeric(circuit.global_phase, instruction="global_phase")
        if phase:
            output.set_global_phase(phase)

        for index, item in enumerate(circuit.data):
            operation = item.operation
            name = operation.name.lower()
            qargs = tuple(circuit.find_bit(qubit).index for qubit in item.qubits)

            condition = getattr(operation, "condition", None)
            if condition is not None:
                raise AdapterConversionError(
                    f"conditional instruction {name!r} at index {index} is not supported"
                )

            if name == "measure":
                if len(qargs) != 1 or len(item.clbits) != 1:
                    raise AdapterConversionError("measure must have one qubit and one clbit")
                measurement_seen = True
                clbit = item.clbits[0]
                classical_index = circuit.find_bit(clbit).index
                output.measure(qargs[0])
                slots.append(
                    MeasurementSlot(
                        qubit_ids[qargs[0]],
                        classical_index,
                        _measurement_key(circuit, clbit),
                    )
                )
                continue

            if name == "barrier":
                output.barrier(list(qargs))
                continue

            if measurement_seen:
                raise AdapterConversionError(
                    f"instruction {name!r} at index {index} occurs after a measurement; "
                    "the Qiskit adapter accepts final measurements only"
                )

            if name == "reset":
                if len(qargs) != 1:
                    raise AdapterConversionError("reset must have one qubit")
                output.reset(qargs[0])
                continue
            if name in {"global_phase", "gphase"} and not qargs:
                output.set_global_phase(_numeric(operation.params[0], instruction=name))
                continue

            params = tuple(_numeric(value, instruction=name) for value in operation.params)
            _append_cqlib_gate(output, name=name, qubits=qargs, params=params)

        output.validate()
        register_sizes = {register.name: register.size for register in circuit.cregs}
        measurements = MeasurementMetadata(
            tuple(slots),
            circuit.num_clbits,
            register_sizes,
        )
        metadata = TranslationMetadata(
            framework="qiskit",
            qubits=qubit_ids,
            measurements=measurements,
            circuit_name=circuit.name,
            parameter_names=tuple(sorted(str(parameter) for parameter in circuit.parameters)),
            global_phase=phase,
            extras={"qiskit_metadata": dict(circuit.metadata or {})},
        )
        return TranslationBundle(cast(CircuitLike, output), metadata)
    except AdapterConversionError:
        raise
    except Exception as exc:
        raise AdapterConversionError(f"failed to translate Qiskit circuit: {exc}") from exc


def compile_qiskit_circuit(
    circuit: QuantumCircuit,
    *,
    device: NormalizedDevice | None = None,
    options: CompilationOptions | None = None,
    compiler: CircuitCompiler | None = None,
    circuit_index: int | None = None,
) -> CompilationArtifact:
    """Translate and compile one Qiskit circuit into validated QCIS."""

    return (compiler or CircuitCompiler()).compile(
        qiskit_to_cqlib(circuit),
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
    qubits = tuple(qubit.index for qubit in operation.qubits)
    return name, qubits, params


def _append_qiskit_gate(
    output: QuantumCircuit,
    *,
    name: str,
    qubits: tuple[int, ...],
    params: tuple[float, ...],
) -> None:
    if name in _SINGLE_NO_PARAM and len(qubits) == 1:
        qiskit_name = "id" if name == "i" else name
        if qiskit_name in {"x2p", "x2m", "y2p", "y2m"}:
            from .gates import X2MGate, X2PGate, Y2MGate, Y2PGate

            gate = {
                "x2p": X2PGate(),
                "x2m": X2MGate(),
                "y2p": Y2PGate(),
                "y2m": Y2MGate(),
            }[qiskit_name]
            output.append(gate, list(qubits))
        else:
            getattr(output, qiskit_name)(qubits[0])
        return
    if name in _SINGLE_ONE_PARAM and len(qubits) == 1 and len(params) == 1:
        if name == "xy":
            output.append(XYGate(params[0]), list(qubits))
        elif name == "xy2p":
            output.append(XY2PGate(params[0]), list(qubits))
        elif name == "xy2m":
            output.append(XY2MGate(params[0]), list(qubits))
        else:
            qiskit_name = "p" if name == "phase" else name
            getattr(output, qiskit_name)(params[0], qubits[0])
        return
    if name == "rxy" and len(qubits) == 1 and len(params) == 2:
        output.append(RXYGate(params[0], params[1]), list(qubits))
        return
    if name == "u" and len(qubits) == 1 and len(params) == 3:
        output.u(params[0], params[1], params[2], qubits[0])
        return
    if name in _TWO_NO_PARAM and len(qubits) == 2:
        getattr(output, name)(qubits[0], qubits[1])
        return
    if name in _TWO_ONE_PARAM and len(qubits) == 2 and len(params) == 1:
        getattr(output, name)(params[0], qubits[0], qubits[1])
        return
    if name == "fsim" and len(qubits) == 2 and len(params) == 2:
        output.append(FSimGate(params[0], params[1]), list(qubits))
        return
    if name == "ccx" and len(qubits) == 3:
        output.ccx(*qubits)
        return
    raise AdapterConversionError(f"unsupported cqlib instruction {name!r} during Qiskit conversion")


def cqlib_to_qiskit(
    circuit: CircuitLike,
    *,
    metadata: TranslationMetadata | None = None,
) -> QuantumCircuit:
    """Convert supported cqlib construction IR back to a Qiskit circuit."""

    circuit.validate()
    physical_qubits = [qubit.index for qubit in circuit.qubits]
    num_qubits = max(physical_qubits, default=-1) + 1
    operations = [_operation_parts(operation) for operation in circuit.operations]
    measurement_count = sum(
        name in {"measure", "measure_bit", "measure_bits"} for name, _, _ in operations
    )
    num_clbits = (
        metadata.measurements.num_classical_bits if metadata is not None else measurement_count
    )

    if metadata is not None and metadata.measurements.register_sizes:
        qreg = QuantumRegister(num_qubits, "q")
        cregs = [
            ClassicalRegister(size, name)
            for name, size in metadata.measurements.register_sizes.items()
        ]
        output = QuantumCircuit(qreg, *cregs, name=metadata.circuit_name)
    else:
        output = QuantumCircuit(
            num_qubits,
            num_clbits,
            name=metadata.circuit_name if metadata is not None else None,
        )

    measurement_index = 0
    slots = metadata.measurements.slots if metadata is not None else ()
    for name, qubits, params in operations:
        if name in {"measure", "measure_bit"}:
            classical = (
                slots[measurement_index].classical_bit
                if measurement_index < len(slots)
                else measurement_index
            )
            output.measure(qubits[0], classical)
            measurement_index += 1
        elif name == "measure_bits":
            for qubit in qubits:
                classical = (
                    slots[measurement_index].classical_bit
                    if measurement_index < len(slots)
                    else measurement_index
                )
                output.measure(qubit, classical)
                measurement_index += 1
        elif name in {"barrier", "b"}:
            output.barrier(*qubits)
        elif name == "reset":
            output.reset(qubits[0])
        elif name == "delay":
            raise AdapterConversionError("cqlib delay conversion is not supported")
        else:
            _append_qiskit_gate(output, name=name, qubits=qubits, params=params)

    try:
        phase = circuit.global_phase
        output.global_phase = phase.evaluate() if hasattr(phase, "evaluate") else float(phase)
    except Exception as exc:
        raise AdapterConversionError(f"failed to convert cqlib global phase: {exc}") from exc
    return output


__all__ = [
    "compile_qiskit_circuit",
    "cqlib_to_qiskit",
    "qiskit_to_cqlib",
]
