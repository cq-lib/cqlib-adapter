"""cqlib compilation, QCIS serialization, and post-compile validation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

from .circuit import CompiledMeasurement, TranslationBundle, TranslationMetadata
from .device import NormalizedDevice, qubit_index
from .errors import AdapterCompileError, ErrorContext
from .options import CompilationMode, CompilationOptions
from .typing import CircuitLike, CompileResultLike, CqlibRuntime, DeviceLike


class DefaultCqlibRuntime:
    """Lazy import facade so the base package remains easy to fake in tests."""

    def normal_mode(self) -> Any:
        return import_module("cqlib.compile").CompileMode.normal()

    def enhanced_mode(self) -> Any:
        return import_module("cqlib.compile").CompileMode.enhanced()

    def compile(
        self,
        circuit: CircuitLike,
        *,
        mode: Any,
        target_basis: list[str] | tuple[str, ...] | None,
        device: DeviceLike | None,
        initial_layout: Any | None,
        resource_policy: Any | None,
        seed: int | None,
    ) -> CompileResultLike:
        cqlib_compile = import_module("cqlib.compile").compile

        return cast(
            CompileResultLike,
            cqlib_compile(
                circuit,
                mode=mode,
                target_basis=target_basis,
                device=device,
                initial_layout=initial_layout,
                resource_policy=resource_policy,
                seed=seed,
            ),
        )

    def dumps(self, circuit: CircuitLike) -> str:
        dumps = import_module("cqlib.ir.qcis").dumps
        return cast(str, dumps(circuit))

    def loads(self, qcis: str) -> CircuitLike:
        loads = import_module("cqlib.ir.qcis").loads
        return cast(CircuitLike, loads(qcis))


@dataclass(frozen=True, slots=True)
class CompilationArtifact:
    """Validated executable and metadata required for result conversion."""

    qcis: str
    circuit: CircuitLike
    metadata: TranslationMetadata
    device: NormalizedDevice | None
    changed: bool
    steps: tuple[Any, ...]
    measurements: tuple[CompiledMeasurement, ...]

    @property
    def num_classical_bits(self) -> int:
        return self.metadata.measurements.num_classical_bits


def _instruction_name(operation: Any) -> str:
    value_instruction = operation.instruction
    instruction = getattr(value_instruction, "instruction", value_instruction)
    if instruction is None:
        return "CLASSICAL_CONTROL"
    name = str(instruction.name).strip().upper()
    # cqlib's construction IR stores measurements as value-producing
    # ``measure_bit``/``measure_bits`` instructions, while QCIS serializes
    # both forms as one or more ``M`` directives.
    return {
        "M": "MEASURE",
        "MEASURE_BIT": "MEASURE",
        "MEASURE_BITS": "MEASURE",
        "B": "BARRIER",
    }.get(name, name)


def _measurement_qubits(circuit: CircuitLike) -> tuple[int, ...]:
    measured: list[int] = []
    for operation in circuit.operations:
        if _instruction_name(operation) == "MEASURE":
            measured.extend(qubit_index(qubit) for qubit in operation.qubits)
    return tuple(measured)


class CircuitCompiler:
    """Compile translated cqlib circuits into validated QCIS executables."""

    _DIRECTIVES = frozenset({"MEASURE", "BARRIER", "RESET", "DELAY", "CLASSICAL_CONTROL"})

    def __init__(self, runtime: CqlibRuntime | None = None) -> None:
        self._runtime = runtime or DefaultCqlibRuntime()

    def compile(
        self,
        bundle: TranslationBundle[CircuitLike],
        *,
        device: NormalizedDevice | None = None,
        options: CompilationOptions | None = None,
        circuit_index: int | None = None,
    ) -> CompilationArtifact:
        options = options or CompilationOptions()
        context = ErrorContext(
            device_name=device.name if device is not None else None,
            circuit_index=circuit_index,
        )
        target_basis = options.target_basis
        if target_basis is None and device is not None:
            target_basis = device.native_gates
        mode = (
            self._runtime.enhanced_mode()
            if options.mode is CompilationMode.ENHANCED
            else self._runtime.normal_mode()
        )
        try:
            bundle.circuit.validate()
            result = self._runtime.compile(
                bundle.circuit,
                mode=mode,
                target_basis=target_basis,
                device=device.cqlib_device if device is not None else None,
                initial_layout=options.initial_layout,
                resource_policy=options.resource_policy,
                seed=options.seed,
            )
            result.circuit.validate()
            self._validate_basis(result.circuit, target_basis)
            if device is not None:
                self._validate_topology(result.circuit, device)
            qcis = self._runtime.dumps(result.circuit).strip()
            if not qcis:
                raise ValueError("cqlib produced empty QCIS")
            reparsed = self._runtime.loads(qcis)
            reparsed.validate()
            # The round-tripped circuit is the exact representation sent to
            # Tianyan. QCIS may expand one construction-IR instruction into
            # multiple executable directives.
            self._validate_basis(reparsed, target_basis)
            if device is not None:
                self._validate_topology(reparsed, device)
            measurements = self._bind_measurements(reparsed, bundle.metadata)
            return CompilationArtifact(
                qcis=qcis,
                circuit=reparsed,
                metadata=bundle.metadata,
                device=device,
                changed=bool(result.changed),
                steps=tuple(result.steps),
                measurements=measurements,
            )
        except AdapterCompileError:
            raise
        except Exception as exc:
            raise AdapterCompileError(str(exc), context=context) from exc

    def validate_qcis(self, qcis: str) -> CircuitLike:
        """Parse and validate externally supplied QCIS through cqlib."""

        if not qcis.strip():
            raise AdapterCompileError("QCIS must not be empty")
        try:
            circuit = self._runtime.loads(qcis)
            circuit.validate()
            return circuit
        except AdapterCompileError:
            raise
        except Exception as exc:
            raise AdapterCompileError(f"invalid QCIS: {exc}") from exc

    def _validate_basis(
        self,
        circuit: CircuitLike,
        target_basis: tuple[str, ...] | None,
    ) -> None:
        if target_basis is None:
            return
        allowed = set(target_basis) | self._DIRECTIVES
        unsupported = sorted(
            {
                name
                for operation in circuit.operations
                if (name := _instruction_name(operation)) not in allowed
            }
        )
        if unsupported:
            raise AdapterCompileError(
                f"compiled circuit contains gates outside target basis: {unsupported}"
            )

    def _validate_topology(
        self,
        circuit: CircuitLike,
        device: NormalizedDevice,
    ) -> None:
        invalid = set(device.invalid_qubits)
        for operation in circuit.operations:
            qubits = tuple(qubit_index(qubit) for qubit in operation.qubits)
            if invalid.intersection(qubits):
                raise AdapterCompileError(
                    f"compiled operation {_instruction_name(operation)} uses invalid qubit"
                )
            if len(qubits) == 2 and not device.supports_coupling(
                qubits[0],
                qubits[1],
                either_direction=True,
            ):
                raise AdapterCompileError(
                    f"compiled two-qubit operation uses unsupported coupling {qubits}"
                )

    def _bind_measurements(
        self,
        circuit: CircuitLike,
        metadata: TranslationMetadata,
    ) -> tuple[CompiledMeasurement, ...]:
        physical_qubits = _measurement_qubits(circuit)
        slots = metadata.measurements.slots
        if len(physical_qubits) != len(slots):
            raise AdapterCompileError(
                "compiled measurement count does not match framework metadata "
                f"({len(physical_qubits)} != {len(slots)})"
            )
        return tuple(
            CompiledMeasurement(
                physical_qubit=physical,
                classical_bit=slot.classical_bit,
                key=slot.key,
            )
            for physical, slot in zip(physical_qubits, slots, strict=True)
        )


__all__ = [
    "CircuitCompiler",
    "CompilationArtifact",
    "DefaultCqlibRuntime",
]
