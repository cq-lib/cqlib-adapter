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

"""Cirq Device exposing one normalized cqlib/Tianyan device."""

from __future__ import annotations

import cirq
import networkx as nx

from cqlib_adapter.common import DeviceStatus, NormalizedDevice, TianyanConnector

from .converter import _is_directly_supported


class TianyanDevice(cirq.Device):
    """Cirq hardware description backed by a normalized cqlib device snapshot."""

    def __init__(
        self,
        device: NormalizedDevice,
        *,
        connector: TianyanConnector | None = None,
    ) -> None:
        self._device = device
        self._connector = connector
        self._qubits = {index: cirq.LineQubit(index) for index in device.qubits}
        graph = nx.DiGraph()
        graph.add_nodes_from(self._qubits.values())
        graph.add_edges_from(
            (self._qubits[edge.source], self._qubits[edge.target]) for edge in device.couplings
        )
        self._metadata = cirq.DeviceMetadata(self._qubits.values(), graph)

    @property
    def metadata(self) -> cirq.DeviceMetadata:
        return self._metadata

    @property
    def normalized_device(self) -> NormalizedDevice:
        return self._device

    @property
    def name(self) -> str:
        return self._device.name

    @property
    def display_name(self) -> str:
        return self._device.display_name

    @property
    def num_qubits(self) -> int:
        return self._device.num_qubits

    @property
    def native_gates(self) -> tuple[str, ...]:
        return self._device.native_gates

    @property
    def status(self) -> DeviceStatus:
        return self._device.status

    @property
    def available(self) -> bool:
        return self._device.available

    @property
    def physical_qubits(self) -> tuple[cirq.LineQubit, ...]:
        return tuple(self._qubits[index] for index in self._device.qubits)

    @classmethod
    def from_connector(cls, connector: TianyanConnector, device_name: str) -> TianyanDevice:
        return cls(connector.resolve_device(device_name), connector=connector)

    def refresh(self) -> NormalizedDevice:
        """Refresh mutable Tianyan status fields."""

        if self._connector is None:
            raise RuntimeError("device refresh requires a TianyanConnector")
        self._device = self._connector.refresh_device_state(self._device)
        return self._device

    def validate_operation(self, operation: cirq.Operation) -> None:
        operation = operation.untagged if isinstance(operation, cirq.TaggedOperation) else operation
        unknown = [qubit for qubit in operation.qubits if qubit not in self.metadata.qubit_set]
        if unknown:
            raise ValueError(
                f"operation uses qubits outside Tianyan device {self.name!r}: {unknown}"
            )
        if not _is_directly_supported(operation):
            raise ValueError(f"unsupported Cirq operation for cqlib compilation: {operation!r}")
        if len(operation.qubits) == 2:
            first = operation.qubits[0]
            second = operation.qubits[1]
            if not isinstance(first, cirq.LineQubit) or not isinstance(second, cirq.LineQubit):
                raise ValueError("Tianyan physical topology requires cirq.LineQubit IDs")
            if not self._device.supports_coupling(first.x, second.x, either_direction=True):
                raise ValueError(
                    f"qubits {first.x} and {second.x} are not coupled on {self.name!r}"
                )
        if len(operation.qubits) > 2:
            raise ValueError("Cirq device validation accepts at most two-qubit native operations")

    def __str__(self) -> str:
        return (
            f"TianyanDevice(name={self.name!r}, qubits={self.num_qubits}, "
            f"status={self.status.value!r})"
        )


__all__ = ["TianyanDevice"]
