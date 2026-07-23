"""Framework-neutral contracts carried through translation and execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from .errors import AdapterConversionError

CircuitT = TypeVar("CircuitT")


def _frozen(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class QubitMapping:
    """Bijective mapping from stable logical IDs to physical qubit indices."""

    logical_to_physical: Mapping[str, int]

    def __post_init__(self) -> None:
        values = tuple(self.logical_to_physical.values())
        if any(not key for key in self.logical_to_physical):
            raise ValueError("logical qubit IDs must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("physical qubit indices must be non-negative integers")
        if len(set(values)) != len(values):
            raise ValueError("logical-to-physical qubit mapping must be bijective")
        object.__setattr__(
            self,
            "logical_to_physical",
            MappingProxyType(dict(self.logical_to_physical)),
        )

    @property
    def physical_to_logical(self) -> Mapping[int, str]:
        return MappingProxyType(
            {physical: logical for logical, physical in self.logical_to_physical.items()}
        )

    def physical(self, logical_qubit: str) -> int:
        try:
            return self.logical_to_physical[logical_qubit]
        except KeyError as exc:
            raise AdapterConversionError(f"unknown logical qubit {logical_qubit!r}") from exc


@dataclass(frozen=True, slots=True)
class MeasurementSlot:
    """One framework measurement destination before physical layout."""

    logical_qubit: str
    classical_bit: int
    key: str | None = None

    def __post_init__(self) -> None:
        if not self.logical_qubit:
            raise ValueError("logical_qubit must not be empty")
        if (
            isinstance(self.classical_bit, bool)
            or not isinstance(self.classical_bit, int)
            or self.classical_bit < 0
        ):
            raise ValueError("classical_bit must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class MeasurementMetadata:
    """Framework classical-bit layout independent of cloud bit ordering."""

    slots: tuple[MeasurementSlot, ...] = ()
    num_classical_bits: int = 0
    register_sizes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bits = tuple(slot.classical_bit for slot in self.slots)
        if len(set(bits)) != len(bits):
            raise ValueError("each classical bit may have at most one final measurement")
        if (
            isinstance(self.num_classical_bits, bool)
            or not isinstance(self.num_classical_bits, int)
            or self.num_classical_bits < 0
        ):
            raise ValueError("num_classical_bits must be non-negative")
        if bits and max(bits) >= self.num_classical_bits:
            raise ValueError("measurement classical bit exceeds num_classical_bits")
        if any(
            not name or isinstance(size, bool) or not isinstance(size, int) or size < 0
            for name, size in self.register_sizes.items()
        ):
            raise ValueError("register names must be non-empty and sizes non-negative")
        if self.register_sizes and sum(self.register_sizes.values()) != self.num_classical_bits:
            raise ValueError("register sizes must sum to num_classical_bits")
        object.__setattr__(self, "register_sizes", _frozen(self.register_sizes))

    @classmethod
    def none(cls) -> MeasurementMetadata:
        return cls()


@dataclass(frozen=True, slots=True)
class TranslationMetadata:
    """Information a framework translator must preserve."""

    framework: str
    qubits: tuple[str, ...]
    measurements: MeasurementMetadata = field(default_factory=MeasurementMetadata.none)
    circuit_name: str | None = None
    parameter_names: tuple[str, ...] = ()
    global_phase: float = 0.0
    warnings: tuple[str, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.framework.strip():
            raise ValueError("framework must not be empty")
        if len(set(self.qubits)) != len(self.qubits) or any(not qubit for qubit in self.qubits):
            raise ValueError("qubit IDs must be non-empty and unique")
        known = set(self.qubits)
        unknown = {
            slot.logical_qubit
            for slot in self.measurements.slots
            if slot.logical_qubit not in known
        }
        if unknown:
            raise ValueError(f"measurements reference unknown logical qubits: {sorted(unknown)}")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter_names must be unique")
        object.__setattr__(self, "framework", self.framework.strip().lower())
        object.__setattr__(self, "extras", _frozen(self.extras))


@dataclass(frozen=True, slots=True)
class TranslationBundle(Generic[CircuitT]):
    """A cqlib circuit paired with framework metadata."""

    circuit: CircuitT
    metadata: TranslationMetadata


@dataclass(frozen=True, slots=True)
class CompiledMeasurement:
    """A classical destination bound to the compiled physical qubit."""

    physical_qubit: int
    classical_bit: int
    key: str | None = None

    def __post_init__(self) -> None:
        indices = (self.physical_qubit, self.classical_bit)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices
        ):
            raise ValueError("compiled measurement indices must be non-negative")


__all__ = [
    "CompiledMeasurement",
    "MeasurementMetadata",
    "MeasurementSlot",
    "QubitMapping",
    "TranslationBundle",
    "TranslationMetadata",
]
