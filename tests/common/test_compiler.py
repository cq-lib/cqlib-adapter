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

from __future__ import annotations

import pytest

from cqlib_adapter.common import (
    AdapterCompileError,
    CircuitCompiler,
    CompilationMode,
    CompilationOptions,
    MeasurementMetadata,
    MeasurementSlot,
    NormalizedDevice,
    TranslationBundle,
    TranslationMetadata,
)

from .fakes import FakeBackend, FakeCircuit, FakeDeviceConfig, FakeRuntime

pytestmark = pytest.mark.unit


def _bundle(circuit: FakeCircuit) -> TranslationBundle[FakeCircuit]:
    metadata = TranslationMetadata(
        framework="qiskit",
        qubits=("q0", "q1"),
        measurements=MeasurementMetadata(
            (MeasurementSlot("q0", 0), MeasurementSlot("q1", 1)),
            num_classical_bits=2,
        ),
    )
    return TranslationBundle(circuit, metadata)


def _three_qubit_bundle(
    circuit: FakeCircuit,
    measurements: MeasurementMetadata,
) -> TranslationBundle[FakeCircuit]:
    return TranslationBundle(
        circuit,
        TranslationMetadata(
            framework="qiskit",
            qubits=("q0", "q1", "q2"),
            measurements=measurements,
        ),
    )


def _device(config: FakeDeviceConfig | None = None) -> NormalizedDevice:
    return NormalizedDevice.from_backend(FakeBackend(config=config))


def test_compile_uses_device_basis_mode_and_roundtrips_qcis() -> None:
    compiled = FakeCircuit(
        (
            ("X90", (0,)),
            ("CZ", (0, 1)),
            ("measure_bit", (1,)),
            ("measure_bit", (0,)),
        )
    )
    runtime = FakeRuntime(compiled)
    artifact = CircuitCompiler(runtime).compile(
        _bundle(FakeCircuit((("H", (0,)),))),
        device=_device(),
        options=CompilationOptions(mode=CompilationMode.ENHANCED, seed=7),
        circuit_index=3,
    )
    assert artifact.qcis.startswith("X90 Q0")
    assert artifact.changed
    assert artifact.steps == ("decompose", "route")
    assert [(item.physical_qubit, item.classical_bit) for item in artifact.measurements] == [
        (1, 0),
        (0, 1),
    ]
    assert runtime.calls[0]["mode"] == "enhanced"
    assert runtime.calls[0]["target_basis"] == ("X90", "CZ")
    assert runtime.calls[0]["device"] is artifact.device.cqlib_device
    assert runtime.calls[0]["seed"] == 7
    assert runtime.loaded == [artifact.qcis]


def test_compile_without_device_uses_explicit_basis() -> None:
    circuit = FakeCircuit((("X", (0,)), ("measure_bit", (0,)), ("measure_bit", (1,))))
    runtime = FakeRuntime(circuit)
    artifact = CircuitCompiler(runtime).compile(
        _bundle(circuit),
        options=CompilationOptions(target_basis=("x",)),
    )
    assert artifact.device is None
    assert runtime.calls[0]["target_basis"] == ("X",)


def test_compile_rejects_gate_outside_target_basis() -> None:
    compiled = FakeCircuit((("H", (0,)), ("measure_bit", (0,)), ("measure_bit", (1,))))
    with pytest.raises(AdapterCompileError, match="outside target basis"):
        CircuitCompiler(FakeRuntime(compiled)).compile(
            _bundle(compiled),
            options=CompilationOptions(target_basis=("X",)),
        )


@pytest.mark.parametrize("gate", ["CZ", "CX"])
def test_compile_rejects_unsupported_coupling_for_two_qubit_gates(gate: str) -> None:
    compiled = FakeCircuit(((gate, (0, 2)), ("measure_bit", (0,)), ("measure_bit", (2,))))
    config = FakeDeviceConfig(gates=("X90", gate), edges=((0, 1), (1, 0)))
    with pytest.raises(AdapterCompileError, match="unsupported coupling"):
        CircuitCompiler(FakeRuntime(compiled)).compile(
            _bundle(compiled),
            device=_device(config),
        )


def test_compile_does_not_require_coupling_for_multiqubit_barrier() -> None:
    compiled = FakeCircuit((("BARRIER", (0, 2)),))
    config = FakeDeviceConfig(edges=((0, 1), (1, 0)))

    artifact = CircuitCompiler(FakeRuntime(compiled)).compile(
        _three_qubit_bundle(compiled, MeasurementMetadata.none()),
        device=_device(config),
    )

    assert artifact.qcis == "BARRIER Q0 Q2"


def test_compile_does_not_require_coupling_for_multiqubit_measurement() -> None:
    compiled = FakeCircuit((("measure_bits", (0, 2)),))
    config = FakeDeviceConfig(edges=((0, 1), (1, 0)))
    measurements = MeasurementMetadata(
        (MeasurementSlot("q0", 0), MeasurementSlot("q2", 1)),
        num_classical_bits=2,
    )

    artifact = CircuitCompiler(FakeRuntime(compiled)).compile(
        _three_qubit_bundle(compiled, measurements),
        device=_device(config),
    )

    assert [(item.physical_qubit, item.classical_bit) for item in artifact.measurements] == [
        (0, 0),
        (2, 1),
    ]


def test_compile_accepts_coupling_in_either_direction_like_cqlib() -> None:
    compiled = FakeCircuit((("CZ", (1, 0)), ("measure_bit", (1,)), ("measure_bit", (0,))))
    config = FakeDeviceConfig(edges=((0, 1),))
    artifact = CircuitCompiler(FakeRuntime(compiled)).compile(
        _bundle(compiled),
        device=_device(config),
    )
    assert artifact.measurements[0].physical_qubit == 1


def test_compile_requires_directed_coupling_for_control_target_gate() -> None:
    compiled = FakeCircuit((("CX", (1, 0)), ("measure_bit", (1,)), ("measure_bit", (0,))))
    config = FakeDeviceConfig(gates=("X90", "CX"), edges=((0, 1),))

    with pytest.raises(AdapterCompileError, match=r"CX uses unsupported coupling \(1, 0\)"):
        CircuitCompiler(FakeRuntime(compiled)).compile(
            _bundle(compiled),
            device=_device(config),
        )


def test_compile_rejects_unclassified_two_qubit_operation() -> None:
    compiled = FakeCircuit(
        (("UNCLASSIFIED2", (0, 1)), ("measure_bit", (0,)), ("measure_bit", (1,)))
    )
    config = FakeDeviceConfig(gates=("X90", "UNCLASSIFIED2"))

    with pytest.raises(AdapterCompileError, match="unclassified two-qubit operation"):
        CircuitCompiler(FakeRuntime(compiled)).compile(
            _bundle(compiled),
            device=_device(config),
        )


def test_compile_rejects_operation_on_invalid_qubit() -> None:
    compiled = FakeCircuit((("X90", (1,)), ("measure_bit", (1,)), ("measure_bit", (0,))))
    config = FakeDeviceConfig(invalid=(1,))
    with pytest.raises(AdapterCompileError, match="invalid qubit"):
        CircuitCompiler(FakeRuntime(compiled)).compile(
            _bundle(compiled),
            device=_device(config),
        )


def test_compile_rejects_measurement_metadata_mismatch() -> None:
    compiled = FakeCircuit((("measure_bit", (0,)),))
    with pytest.raises(AdapterCompileError, match="measurement count"):
        CircuitCompiler(FakeRuntime(compiled)).compile(
            _bundle(compiled),
            options=CompilationOptions(target_basis=("X",)),
        )


def test_compile_rejects_empty_serialization() -> None:
    compiled = FakeCircuit(())
    metadata = TranslationMetadata("cirq", (), MeasurementMetadata.none())
    with pytest.raises(AdapterCompileError, match="empty QCIS"):
        CircuitCompiler(FakeRuntime(compiled)).compile(TranslationBundle(compiled, metadata))


def test_compile_wraps_source_validation_with_context() -> None:
    circuit = FakeCircuit((), fail_validation=True)
    metadata = TranslationMetadata("cirq", ())
    with pytest.raises(AdapterCompileError) as raised:
        CircuitCompiler(FakeRuntime()).compile(
            TranslationBundle(circuit, metadata),
            circuit_index=4,
        )
    assert "circuit_index=4" in str(raised.value)
    assert isinstance(raised.value.__cause__, ValueError)


def test_validate_qcis_rejects_empty_and_invalid_input() -> None:
    compiler = CircuitCompiler(FakeRuntime())
    with pytest.raises(AdapterCompileError, match="must not be empty"):
        compiler.validate_qcis(" ")

    invalid_runtime = FakeRuntime(FakeCircuit((), fail_validation=True))
    with pytest.raises(AdapterCompileError, match="invalid QCIS"):
        CircuitCompiler(invalid_runtime).validate_qcis("X Q0")
