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

"""CUDA-Q-facing Tianyan target and device information."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cqlib_adapter.common import NormalizedDevice, TianyanConnector


@dataclass(slots=True)
class TianyanTarget:
    """Stable target metadata for a normalized cqlib-tianyan device.

    CUDA-Q does not provide a public constructor for third-party ``Target``
    objects, so this adapter exposes its own snapshot instead of modifying the
    CUDA-Q global target registry.
    """

    _device: NormalizedDevice
    _connector: TianyanConnector | None = None

    @property
    def device(self) -> NormalizedDevice:
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
    def qubits(self) -> tuple[int, ...]:
        return self._device.qubits

    @property
    def native_gates(self) -> tuple[str, ...]:
        return self._device.native_gates

    @property
    def couplings(self) -> tuple[tuple[int, int], ...]:
        return tuple((edge.source, edge.target) for edge in self._device.couplings)

    @property
    def usable_qubits(self) -> tuple[int, ...]:
        return self._device.usable_qubits

    @property
    def invalid_qubits(self) -> tuple[int, ...]:
        return self._device.invalid_qubits

    @property
    def available(self) -> bool:
        return self._device.available

    @property
    def status(self) -> str:
        return self._device.status.value

    @property
    def toll(self) -> str:
        return self._device.toll.value

    def supports(self, gate: str) -> bool:
        return self._device.supports(gate)

    def supports_coupling(
        self, source: int, target: int, *, either_direction: bool = False
    ) -> bool:
        return self._device.supports_coupling(
            source,
            target,
            either_direction=either_direction,
        )

    def refresh(self) -> NormalizedDevice:
        """Refresh status and availability when backed by a connector."""

        if self._connector is not None:
            self._device = self._connector.refresh_device_state(self._device)
        return self._device

    def as_dict(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(
            {
                "name": self.name,
                "display_name": self.display_name,
                "num_qubits": self.num_qubits,
                "qubits": self.qubits,
                "native_gates": self.native_gates,
                "couplings": self.couplings,
                "usable_qubits": self.usable_qubits,
                "invalid_qubits": self.invalid_qubits,
                "available": self.available,
                "status": self.status,
                "toll": self.toll,
                "properties_complete": self._device.properties_complete,
            }
        )


def target_from_device(
    device: NormalizedDevice,
    *,
    connector: TianyanConnector | None = None,
) -> TianyanTarget:
    """Create a CUDA-Q-facing target snapshot from normalized device data."""

    return TianyanTarget(device, connector)


__all__ = ["TianyanTarget", "target_from_device"]
