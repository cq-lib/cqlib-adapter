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

"""Build Qiskit Target objects from normalized Tianyan devices."""

from __future__ import annotations

from typing import Any

from qiskit.circuit import Barrier, Delay, Measure, Parameter, Reset
from qiskit.transpiler import CouplingMap, Target

from cqlib_adapter.common import AdapterDeviceError, NormalizedDevice

from .gates import qcis_instruction

_ALIASES = {
    "ID": "I",
    "P": "PHASE",
}
_MEASURE_NAMES = {"M", "MEASURE", "MEASURE_BIT", "MEASURE_BITS"}
_BARRIER_NAMES = {"B", "BARRIER"}
_SYMMETRIC_TWO_QUBIT = {"CZ", "SWAP", "RXX", "RYY", "RZZ", "FSIM"}


def _require_dense_qubits(device: NormalizedDevice) -> None:
    expected = tuple(range(device.num_qubits))
    if device.qubits != expected:
        raise AdapterDeviceError(
            "Qiskit Target requires zero-based dense physical qubit IDs; "
            f"device {device.name!r} advertises {device.qubits}"
        )


def coupling_map_from_device(device: NormalizedDevice) -> CouplingMap:
    """Create the Qiskit coupling map advertised by a Tianyan device."""

    _require_dense_qubits(device)
    return CouplingMap([[edge.source, edge.target] for edge in device.couplings])


def _properties_for_gate(
    device: NormalizedDevice,
    gate: Any,
) -> dict[tuple[int, ...] | None, None]:
    num_qubits = int(gate.num_qubits)
    if num_qubits == 1:
        return {(qubit,): None for qubit in device.usable_qubits}
    if num_qubits == 2:
        edges = {(edge.source, edge.target) for edge in device.couplings}
        name = str(gate.name).upper()
        if name in _SYMMETRIC_TWO_QUBIT:
            edges |= {(target, source) for source, target in edges}
        if not edges:
            return {None: None}
        return {edge: None for edge in sorted(edges)}
    return {None: None}


def target_from_device(device: NormalizedDevice) -> Target:
    """Expose native gates, qubits and topology through a Qiskit Target."""

    _require_dense_qubits(device)
    target = Target(
        description=f"{device.display_name} ({device.name})",
        num_qubits=device.num_qubits,
    )
    added: set[str] = set()
    wants_barrier = False
    wants_reset = False
    wants_delay = False

    for raw_name in device.native_gates:
        normalized = _ALIASES.get(raw_name.strip().upper(), raw_name.strip().upper())
        if normalized in _MEASURE_NAMES:
            continue
        if normalized in _BARRIER_NAMES:
            wants_barrier = True
            continue
        if normalized == "RESET":
            wants_reset = True
            continue
        if normalized == "DELAY":
            wants_delay = True
            continue
        if normalized in {"GPHASE", "GLOBAL_PHASE"}:
            continue
        try:
            instruction = qcis_instruction(normalized)
        except KeyError as exc:
            raise AdapterDeviceError(
                f"device {device.name!r} advertises unsupported native gate {raw_name!r}"
            ) from exc
        if instruction.name in added:
            continue
        target.add_instruction(
            instruction,
            _properties_for_gate(device, instruction),
        )
        added.add(instruction.name)

    one_qubit_properties = {(qubit,): None for qubit in device.usable_qubits}
    # Measurement is required for the BackendV2 execution contract even when
    # cqlib reports it as a directive instead of a native unitary gate.
    if "measure" not in added:
        target.add_instruction(Measure(), one_qubit_properties)
        added.add("measure")
    if wants_reset and "reset" not in added:
        target.add_instruction(Reset(), one_qubit_properties)
    if wants_delay and "delay" not in added:
        target.add_instruction(Delay(Parameter("duration")), one_qubit_properties)
    if wants_barrier or "barrier" not in added:
        target.add_instruction(Barrier, name="barrier")

    return target


__all__ = ["coupling_map_from_device", "target_from_device"]
