"""Normalized Tianyan device information consumed by four frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .errors import AdapterDeviceError, ErrorContext
from .typing import BackendLike, DeviceLike, QubitLike


def qubit_index(qubit: QubitLike | int) -> int:
    """Return a physical index from a cqlib Qubit or an integer."""

    value = qubit if isinstance(qubit, int) else qubit.index
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdapterDeviceError(f"invalid physical qubit index {value!r}")
    return value


def _enum_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw).strip().lower()
    return text.rsplit(".", 1)[-1]


class DeviceStatus(StrEnum):
    RUNNING = "running"
    CALIBRATION = "calibration"
    UNDER_MAINTENANCE = "under_maintenance"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: Any) -> DeviceStatus:
        text = _enum_text(value)
        aliases = {
            "online": cls.RUNNING,
            "available": cls.RUNNING,
            "unavailable": cls.OFFLINE,
            "maintenance": cls.UNDER_MAINTENANCE,
            "maintain": cls.UNDER_MAINTENANCE,
        }
        try:
            return cls(text)
        except ValueError:
            return aliases.get(text, cls.UNKNOWN)


class DeviceToll(StrEnum):
    FREE = "free"
    PAID = "paid"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: Any) -> DeviceToll:
        text = _enum_text(value)
        aliases = {"charge": cls.PAID, "charged": cls.PAID}
        try:
            return cls(text)
        except ValueError:
            return aliases.get(text, cls.UNKNOWN)


@dataclass(frozen=True, order=True, slots=True)
class Coupling:
    """One directed physical coupling advertised by cqlib."""

    source: int
    target: int

    def __post_init__(self) -> None:
        indices = (self.source, self.target)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices
        ):
            raise ValueError("coupling indices must be non-negative")
        if self.source == self.target:
            raise ValueError("a coupling cannot be a self-loop")


@dataclass(frozen=True, slots=True)
class NormalizedDevice:
    """Stable device snapshot independent of framework and Tianyan enums."""

    name: str
    display_name: str
    num_qubits: int
    qubits: tuple[int, ...]
    native_gates: tuple[str, ...]
    couplings: tuple[Coupling, ...]
    usable_qubits: tuple[int, ...]
    invalid_qubits: tuple[int, ...]
    status: DeviceStatus
    toll: DeviceToll
    available: bool
    cqlib_device: DeviceLike
    properties_complete: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.num_qubits <= 0:
            raise ValueError("device name must be non-empty and num_qubits must be positive")
        all_qubits = set(self.qubits)
        if len(all_qubits) != self.num_qubits or len(all_qubits) != len(self.qubits):
            raise ValueError("qubits must contain num_qubits unique physical IDs")
        usable = set(self.usable_qubits)
        invalid = set(self.invalid_qubits)
        if usable & invalid or usable | invalid != all_qubits:
            raise ValueError("usable and invalid qubits must partition the device")
        if any(
            edge.source not in all_qubits or edge.target not in all_qubits
            for edge in self.couplings
        ):
            raise ValueError("coupling references a qubit outside the device")
        gates = tuple(
            dict.fromkeys(gate.strip().upper() for gate in self.native_gates if gate.strip())
        )
        if not gates:
            raise ValueError("device must advertise at least one native gate")
        object.__setattr__(self, "native_gates", gates)
        object.__setattr__(self, "qubits", tuple(sorted(all_qubits)))
        object.__setattr__(self, "couplings", tuple(sorted(set(self.couplings))))
        object.__setattr__(self, "usable_qubits", tuple(sorted(usable)))
        object.__setattr__(self, "invalid_qubits", tuple(sorted(invalid)))

    @classmethod
    def from_backend(cls, backend: BackendLike) -> NormalizedDevice:
        """Normalize a cqlib-tianyan backend without importing framework packages."""

        context = ErrorContext(device_name=getattr(backend, "name", None))
        try:
            config = backend.device_config()
            num_qubits = backend.num_qubits()
            qubits = tuple(qubit_index(qubit) for qubit in config.qubits)
            if len(qubits) != num_qubits:
                raise ValueError("backend qubit count disagrees with cqlib device configuration")
            invalid = tuple(qubit_index(qubit) for qubit in config.invalid_qubits)
            usable = tuple(sorted(set(qubits) - set(invalid)))
            couplings: set[Coupling] = set()
            for source_qubit in config.topology.qubits:
                source = qubit_index(source_qubit)
                for target_qubit in config.topology.successors(source_qubit):
                    couplings.add(Coupling(source, qubit_index(target_qubit)))
            native_gates = tuple(str(getattr(gate, "name", gate)) for gate in config.native_gates)
            return cls(
                name=backend.name,
                display_name=backend.display_name or backend.name,
                num_qubits=num_qubits,
                qubits=qubits,
                native_gates=native_gates,
                couplings=tuple(couplings),
                usable_qubits=usable,
                invalid_qubits=invalid,
                status=DeviceStatus.parse(backend.status),
                toll=DeviceToll.parse(backend.toll),
                available=bool(backend.is_available()),
                cqlib_device=config,
                properties_complete=False,
            )
        except AdapterDeviceError:
            raise
        except Exception as exc:
            raise AdapterDeviceError(
                f"failed to normalize Tianyan backend: {exc}",
                context=context,
            ) from exc

    def supports(self, gate: str) -> bool:
        return gate.strip().upper() in self.native_gates

    def supports_coupling(
        self, source: int, target: int, *, either_direction: bool = False
    ) -> bool:
        edge = Coupling(source, target)
        if edge in self.couplings:
            return True
        return either_direction and Coupling(target, source) in self.couplings


__all__ = [
    "Coupling",
    "DeviceStatus",
    "DeviceToll",
    "NormalizedDevice",
    "qubit_index",
]
