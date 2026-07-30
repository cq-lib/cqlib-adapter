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

"""Conversion between Cirq circuits and the cqlib construction IR."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from importlib import import_module
from math import pi
from typing import Any, cast

import cirq
import numpy as np

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

from .gates import (
    RXYGate,
    X2MGate,
    X2PGate,
    XY2MGate,
    XY2PGate,
    XYGate,
    Y2MGate,
    Y2PGate,
)


def _numeric(value: Any, *, operation: str) -> float:
    if cirq.is_parameterized(value):
        raise AdapterConversionError(
            f"Cirq operation {operation!r} has an unbound parameter {value!r}; "
            "resolve parameters or execute it through TianyanSampler.run_sweep"
        )
    try:
        result = complex(value)
    except (TypeError, ValueError) as exc:
        raise AdapterConversionError(
            f"Cirq operation {operation!r} has a non-numeric parameter {value!r}"
        ) from exc
    if abs(result.imag) > 1e-12:
        raise AdapterConversionError(
            f"Cirq operation {operation!r} requires a real parameter, got {value!r}"
        )
    return float(result.real)


def _ordered_qubits(
    circuit: cirq.AbstractCircuit,
    qubit_order: cirq.QubitOrderOrList | None,
) -> tuple[cirq.Qid, ...]:
    qubits = circuit.all_qubits()
    if not qubits:
        raise AdapterConversionError("Cirq circuit must contain at least one qubit")
    order = cirq.QubitOrder.as_qubit_order(
        cirq.QubitOrder.DEFAULT if qubit_order is None else qubit_order
    ).order_for(qubits)
    if len(order) != len(qubits) or set(order) != set(qubits):
        raise AdapterConversionError(
            "Cirq qubit_order must contain every circuit qubit exactly once"
        )
    return tuple(order)


def _unwrap(operation: cirq.Operation) -> cirq.Operation:
    return operation.untagged if isinstance(operation, cirq.TaggedOperation) else operation


def _append_cqlib_gate(output: Any, operation: cirq.Operation, qubits: tuple[int, ...]) -> None:
    operation = _unwrap(operation)
    gate = operation.gate
    if gate is None:
        raise AdapterConversionError(f"unsupported gate-less Cirq operation {operation!r}")
    name = type(gate).__name__

    fixed = {
        X2PGate: "x2p",
        X2MGate: "x2m",
        Y2PGate: "y2p",
        Y2MGate: "y2m",
    }
    for gate_type, method in fixed.items():
        if isinstance(gate, gate_type):
            getattr(output, method)(qubits[0])
            return
    if isinstance(gate, XYGate):
        output.xy(qubits[0], _numeric(gate.theta, operation=name))
        return
    if isinstance(gate, XY2PGate):
        output.xy2p(qubits[0], _numeric(gate.phi, operation=name))
        return
    if isinstance(gate, XY2MGate):
        output.xy2m(qubits[0], _numeric(gate.phi, operation=name))
        return
    if isinstance(gate, RXYGate):
        output.rxy(
            qubits[0],
            _numeric(gate.theta, operation=name),
            _numeric(gate.phi, operation=name),
        )
        return
    if isinstance(gate, cirq.FSimGate):
        output.fsim(
            qubits[0],
            qubits[1],
            _numeric(gate.theta, operation=name),
            _numeric(gate.phi, operation=name),
        )
        return
    if isinstance(gate, cirq.IdentityGate):
        for qubit in qubits:
            output.i(qubit)
        return
    if isinstance(gate, cirq.HPowGate):
        exponent = _numeric(gate.exponent, operation=name)
        if not np.isclose(exponent % 2, 1):
            raise AdapterConversionError("only odd-integer powers of Cirq H are supported")
        output.h(qubits[0])
        return
    if isinstance(gate, cirq.XPowGate):
        exponent = _numeric(gate.exponent, operation=name)
        angle = pi * exponent
        if np.isclose(exponent, 1):
            output.x(qubits[0])
        elif np.isclose(exponent, 0.5):
            output.x2p(qubits[0])
        elif np.isclose(exponent, -0.5):
            output.x2m(qubits[0])
        else:
            output.rx(qubits[0], angle)
        return
    if isinstance(gate, cirq.YPowGate):
        exponent = _numeric(gate.exponent, operation=name)
        angle = pi * exponent
        if np.isclose(exponent, 1):
            output.y(qubits[0])
        elif np.isclose(exponent, 0.5):
            output.y2p(qubits[0])
        elif np.isclose(exponent, -0.5):
            output.y2m(qubits[0])
        else:
            output.ry(qubits[0], angle)
        return
    if isinstance(gate, cirq.ZPowGate):
        output.rz(qubits[0], pi * _numeric(gate.exponent, operation=name))
        return
    if isinstance(gate, cirq.PhasedXPowGate):
        output.rxy(
            qubits[0],
            pi * _numeric(gate.exponent, operation=name),
            pi * _numeric(gate.phase_exponent, operation=name),
        )
        return
    if isinstance(gate, cirq.CNotPowGate):
        exponent = _numeric(gate.exponent, operation=name)
        if not np.isclose(exponent % 2, 1):
            raise AdapterConversionError("only odd-integer powers of Cirq CNOT are supported")
        output.cx(*qubits)
        return
    if isinstance(gate, cirq.CZPowGate):
        exponent = _numeric(gate.exponent, operation=name)
        if not np.isclose(exponent % 2, 1):
            raise AdapterConversionError("only odd-integer powers of Cirq CZ are supported")
        output.cz(*qubits)
        return
    if isinstance(gate, cirq.SwapPowGate):
        exponent = _numeric(gate.exponent, operation=name)
        if not np.isclose(exponent % 2, 1):
            raise AdapterConversionError("only odd-integer powers of Cirq SWAP are supported")
        output.swap(*qubits)
        return
    if isinstance(gate, cirq.CCXPowGate):
        exponent = _numeric(gate.exponent, operation=name)
        if not np.isclose(exponent % 2, 1):
            raise AdapterConversionError("only odd-integer powers of Cirq TOFFOLI are supported")
        output.ccx(*qubits)
        return
    raise AdapterConversionError(f"unsupported Cirq gate {name!r}")


def _is_directly_supported(operation: cirq.Operation) -> bool:
    operation = _unwrap(operation)
    gate = operation.gate
    return isinstance(
        gate,
        cirq.MeasurementGate
        | cirq.GlobalPhaseGate
        | cirq.IdentityGate
        | cirq.HPowGate
        | cirq.XPowGate
        | cirq.YPowGate
        | cirq.ZPowGate
        | cirq.PhasedXPowGate
        | cirq.CNotPowGate
        | cirq.CZPowGate
        | cirq.SwapPowGate
        | cirq.CCXPowGate
        | cirq.FSimGate
        | X2PGate
        | X2MGate
        | Y2PGate
        | Y2MGate
        | XYGate
        | XY2PGate
        | XY2MGate
        | RXYGate,
    )


def decompose_cirq_circuit(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Expand composite operations while preserving supported native gates."""

    output = cirq.Circuit()
    for operation in circuit.all_operations():
        try:
            expanded = cirq.decompose(
                operation,
                keep=_is_directly_supported,
                on_stuck_raise=lambda stuck: AdapterConversionError(
                    f"cannot decompose unsupported Cirq operation {stuck!r}"
                ),
            )
            for item in expanded:
                output.append(item, strategy=cirq.InsertStrategy.NEW)
        except AdapterConversionError:
            raise
        except Exception as exc:
            raise AdapterConversionError(
                f"failed to decompose Cirq operation {operation!r}: {exc}"
            ) from exc
    return output


def cirq_to_cqlib(
    circuit: cirq.AbstractCircuit,
    *,
    qubit_order: cirq.QubitOrderOrList | None = None,
) -> TranslationBundle[CircuitLike]:
    """Translate a fully resolved Cirq circuit into real cqlib IR."""

    if not isinstance(circuit, cirq.AbstractCircuit):
        raise TypeError("circuit must be a cirq.AbstractCircuit")
    parameter_names = tuple(sorted(cirq.parameter_names(circuit)))
    if parameter_names:
        raise AdapterConversionError(
            f"Cirq circuit has unbound parameters {list(parameter_names)!r}; "
            "use cirq.resolve_parameters or TianyanSampler.run_sweep"
        )
    order = _ordered_qubits(circuit, qubit_order)
    decomposed = decompose_cirq_circuit(circuit)
    wire_map = {qubit: index for index, qubit in enumerate(order)}
    qubit_ids = tuple(f"qid{index}" for index in range(len(order)))
    output = import_module("cqlib").Circuit(len(order))
    slots: list[MeasurementSlot] = []
    key_bits: dict[str, list[int]] = defaultdict(list)
    seen_keys: set[str] = set()
    measured_qubits: set[cirq.Qid] = set()
    measurement_seen = False
    global_phase = 0.0

    try:
        for index, raw_operation in enumerate(decomposed.all_operations()):
            operation = _unwrap(raw_operation)
            gate = operation.gate
            if isinstance(gate, cirq.GlobalPhaseGate):
                if measurement_seen:
                    raise AdapterConversionError(
                        f"global phase at operation index {index} occurs after measurement"
                    )
                phase = float(np.angle(complex(gate.coefficient)))
                global_phase += phase
                output.set_global_phase(global_phase)
                continue
            if isinstance(gate, cirq.MeasurementGate):
                measurement_seen = True
                key = cirq.measurement_key_name(operation)
                if key in seen_keys:
                    raise AdapterConversionError(
                        f"repeated Cirq measurement key {key!r} is not supported"
                    )
                seen_keys.add(key)
                confusion_map = getattr(gate, "confusion_map", None)
                if confusion_map:
                    raise AdapterConversionError("Cirq measurement confusion_map is not supported")
                mask = tuple(gate.full_invert_mask())
                for position, qubit in enumerate(operation.qubits):
                    if qubit in measured_qubits:
                        raise AdapterConversionError(
                            f"Cirq qubit {qubit!r} is measured more than once"
                        )
                    measured_qubits.add(qubit)
                    logical = wire_map[qubit]
                    if mask[position]:
                        output.x(logical)
                    classical = len(slots)
                    output.measure(logical)
                    slots.append(MeasurementSlot(qubit_ids[logical], classical, key))
                    key_bits[key].append(classical)
                continue
            if measurement_seen:
                raise AdapterConversionError(
                    f"Cirq operation {operation!r} at index {index} occurs after measurement; "
                    "the Cirq adapter accepts final measurements only"
                )
            qubits = tuple(wire_map[qubit] for qubit in operation.qubits)
            _append_cqlib_gate(output, operation, qubits)

        if not slots:
            raise AdapterConversionError("Cirq sampler circuit must contain a final measurement")
        output.validate()
        measurements = MeasurementMetadata(
            tuple(slots),
            len(slots),
            {key: len(bits) for key, bits in key_bits.items()},
        )
        metadata = TranslationMetadata(
            framework="cirq",
            qubits=qubit_ids,
            measurements=measurements,
            parameter_names=parameter_names,
            global_phase=global_phase,
            extras={
                "cirq_qubit_order": order,
                "measurement_key_bits": tuple((key, tuple(bits)) for key, bits in key_bits.items()),
            },
        )
        return TranslationBundle(cast(CircuitLike, output), metadata)
    except AdapterConversionError:
        raise
    except Exception as exc:
        raise AdapterConversionError(f"failed to translate Cirq circuit: {exc}") from exc


def compile_cirq_circuit(
    circuit: cirq.AbstractCircuit,
    *,
    qubit_order: cirq.QubitOrderOrList | None = None,
    device: NormalizedDevice | None = None,
    options: CompilationOptions | None = None,
    compiler: CircuitCompiler | None = None,
    circuit_index: int | None = None,
) -> CompilationArtifact:
    """Translate and compile one Cirq circuit into validated QCIS."""

    return (compiler or CircuitCompiler()).compile(
        cirq_to_cqlib(circuit, qubit_order=qubit_order),
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


def _cqlib_gate(name: str, params: tuple[float, ...]) -> cirq.Gate:
    no_param: dict[str, cirq.Gate] = {
        "i": cirq.I,
        "h": cirq.H,
        "x": cirq.X,
        "y": cirq.Y,
        "z": cirq.Z,
        "s": cirq.S,
        "sdg": cirq.S**-1,
        "t": cirq.T,
        "tdg": cirq.T**-1,
        "x2p": X2PGate(),
        "x2m": X2MGate(),
        "y2p": Y2PGate(),
        "y2m": Y2MGate(),
        "cx": cirq.CNOT,
        "cy": cirq.Y.controlled(),
        "cz": cirq.CZ,
        "swap": cirq.SWAP,
        "ccx": cirq.TOFFOLI,
    }
    if name in no_param and not params:
        return no_param[name]
    one_param = {
        "rx": cirq.rx,
        "ry": cirq.ry,
        "rz": cirq.rz,
        "xy": XYGate,
        "xy2p": XY2PGate,
        "xy2m": XY2MGate,
    }
    if name in one_param and len(params) == 1:
        return one_param[name](params[0])
    if name == "rxy" and len(params) == 2:
        return RXYGate(*params)
    if name == "fsim" and len(params) == 2:
        return cirq.FSimGate(*params)
    raise AdapterConversionError(f"unsupported cqlib instruction {name!r} during Cirq conversion")


def cqlib_to_cirq(
    circuit: CircuitLike,
    *,
    metadata: TranslationMetadata | None = None,
) -> cirq.Circuit:
    """Convert supported cqlib construction IR back to a Cirq circuit."""

    circuit.validate()
    physical = tuple(qubit.index for qubit in circuit.qubits)
    width = max(physical, default=-1) + 1
    stored_order = metadata.extras.get("cirq_qubit_order") if metadata is not None else None
    if stored_order is not None and len(stored_order) >= width:
        qubits: Sequence[cirq.Qid] = tuple(stored_order)
    else:
        qubits = tuple(cirq.LineQubit.range(width))
    output = cirq.Circuit()
    try:
        phase_value = circuit.global_phase
        phase = (
            float(phase_value.evaluate())
            if hasattr(phase_value, "evaluate")
            else float(phase_value)
        )
    except Exception as exc:
        raise AdapterConversionError(f"failed to convert cqlib global phase: {exc}") from exc
    if not np.isclose(phase, 0.0):
        output.append(cirq.global_phase_operation(np.exp(1j * phase)))
    measured: list[tuple[int, int]] = []
    measurement_index = 0
    slots = metadata.measurements.slots if metadata is not None else ()

    for operation in circuit.operations:
        name, indices, params = _operation_parts(operation)
        if name in {"measure", "measure_bit"}:
            classical = (
                slots[measurement_index].classical_bit
                if measurement_index < len(slots)
                else measurement_index
            )
            measured.append((indices[0], classical))
            measurement_index += 1
        elif name == "measure_bits":
            for index in indices:
                classical = (
                    slots[measurement_index].classical_bit
                    if measurement_index < len(slots)
                    else measurement_index
                )
                measured.append((index, classical))
                measurement_index += 1
        elif name in {"barrier", "b"}:
            continue
        else:
            output.append(_cqlib_gate(name, params).on(*(qubits[index] for index in indices)))

    by_key: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for physical_index, classical in measured:
        key = slots[classical].key if classical < len(slots) else "m"
        by_key[key or "m"].append((classical, physical_index))
    for key, values in by_key.items():
        ordered = [qubits[physical_index] for _, physical_index in sorted(values)]
        output.append(cirq.measure(*ordered, key=key))
    return output


__all__ = [
    "cirq_to_cqlib",
    "compile_cirq_circuit",
    "cqlib_to_cirq",
    "decompose_cirq_circuit",
]
