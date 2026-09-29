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

from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import requires, version
from pathlib import Path

import numpy as np
import pytest
from cqlib import Circuit, Qubit, Statevector
from cqlib.circuit import Instruction, StandardGate
from cqlib.compile.resource import ResourcePolicy
from cqlib.device import Device, ExecutionResult, Layout, Topology
from packaging.requirements import Requirement

from cqlib_adapter.common import (
    AdapterCompileError,
    CircuitCompiler,
    CompilationMode,
    CompilationOptions,
    MeasurementMetadata,
    MeasurementSlot,
    NormalizedDevice,
    ResultConverter,
    RunOptions,
    TianyanConnector,
    TranslationBundle,
    TranslationMetadata,
    qubit_index,
)
from cqlib_adapter.common.compiler import DefaultCqlibRuntime

pytestmark = pytest.mark.integration


def _instruction(gate: StandardGate) -> Instruction:
    return Instruction.from_standard_gate(gate)


def _native_basis() -> tuple[str, ...]:
    return ("RZ", "X2P", "X2M", "Y2P", "Y2M", "XY2P", "XY2M", "CZ", "GPHASE")


def _native_instructions() -> list[Instruction]:
    return [
        _instruction(StandardGate.RZ),
        _instruction(StandardGate.X2P),
        _instruction(StandardGate.X2M),
        _instruction(StandardGate.Y2P),
        _instruction(StandardGate.Y2M),
        _instruction(StandardGate.XY2P),
        _instruction(StandardGate.XY2M),
        _instruction(StandardGate.CZ),
        _instruction(StandardGate.GPhase),
    ]


def _operation_name(operation: object) -> str:
    value_instruction = operation.instruction  # type: ignore[attr-defined]
    instruction = value_instruction.instruction
    name = instruction.name.upper()
    if name in {"MEASURE_BIT", "MEASURE_BITS"}:
        return "MEASURE"
    return name


def _unitary_matrix(circuit: Circuit, num_qubits: int) -> np.ndarray:
    columns = []
    for basis_state in range(2**num_qubits):
        prepared = Circuit(num_qubits)
        for qubit in range(num_qubits):
            if basis_state & (1 << qubit):
                prepared.x(qubit)
        for operation in circuit.operations:
            prepared.append(operation)
        columns.append(Statevector.from_circuit(prepared).data)
    return np.array(columns).T


class _NativeBackend:
    def __init__(self, config: Device, *, name: str = "native-device") -> None:
        self.name = name
        self.display_name = "Native cqlib device"
        self.status = "running"
        self.toll = "free"
        self._config = config

    def device_config(self) -> Device:
        return self._config

    def num_qubits(self) -> int:
        return len(self._config.qubits)

    def is_available(self) -> bool:
        return True


class _NativeHandle:
    def __init__(self, results: list[ExecutionResult], *, shots: int) -> None:
        self.task_ids = [result.task_id for result in results]
        self.device_name = "native-device"
        self.shots = shots
        self._results = results

    def status(self) -> list[ExecutionResult]:
        return []

    def wait(
        self,
        timeout: float | None = None,
        poll_interval: float = 5.0,
    ) -> list[ExecutionResult]:
        assert timeout is not None
        assert timeout > 0
        assert poll_interval > 0
        return list(self._results)


class _ExecutingNativeBackend(_NativeBackend):
    def __init__(self, config: Device, result: ExecutionResult) -> None:
        super().__init__(config)
        self._result = result
        self.submissions: list[tuple[list[str], int]] = []

    def run(self, circuits: list[str], shots: int = 1024) -> _NativeHandle:
        self.submissions.append((list(circuits), shots))
        return _NativeHandle([self._result], shots=shots)


class _NativePlatform:
    def __init__(self, backend: _NativeBackend) -> None:
        self._backend = backend

    def list_backends(self) -> list[_NativeBackend]:
        return [self._backend]

    def get_backend(self, name: str) -> _NativeBackend:
        if name != self._backend.name:
            raise KeyError(name)
        return self._backend


def _metadata(
    qubits: tuple[str, ...],
    slots: tuple[MeasurementSlot, ...],
    num_classical_bits: int,
) -> TranslationMetadata:
    return TranslationMetadata(
        "qiskit",
        qubits,
        MeasurementMetadata(slots, num_classical_bits),
    )


def test_uses_local_rust_compiled_cqlib_extension() -> None:
    import cqlib
    import cqlib._native as native

    requirement = next(
        requirement
        for requirement in (Requirement(item) for item in requires("cqlib-adapter") or [])
        if requirement.name == "cqlib"
    )
    assert requirement.specifier.contains(version("cqlib"), prereleases=True)
    assert Path(cqlib.__file__).resolve().is_file()
    native_path = str(Path(native.__file__).resolve())
    assert any(native_path.endswith(suffix) for suffix in EXTENSION_SUFFIXES)


def test_default_runtime_exposes_real_modes_compile_and_qcis_roundtrip() -> None:
    runtime = DefaultCqlibRuntime()
    assert runtime.normal_mode() == runtime.normal_mode()
    assert runtime.enhanced_mode() == runtime.enhanced_mode()
    assert runtime.normal_mode() != runtime.enhanced_mode()

    circuit = Circuit(3)
    circuit.x2p(0)
    circuit.x2m(1)
    circuit.y2p(2)
    circuit.y2m(0)
    circuit.xy2p(1, 0.25)
    circuit.xy2m(2, -0.5)
    circuit.rz(0, 0.125)
    circuit.cz(0, 1)
    circuit.barrier([0, 1, 2])
    circuit.measure_bits([0, 1, 2])

    qcis = runtime.dumps(circuit)
    assert "X2P Q0" in qcis
    assert "XY2M Q2" in qcis
    assert "CZ Q0 Q1" in qcis
    assert "M Q0 Q1 Q2" in qcis

    reparsed = runtime.loads(qcis)
    reparsed.validate()
    assert [_operation_name(operation) for operation in reparsed.operations][-3:] == [
        "MEASURE",
        "MEASURE",
        "MEASURE",
    ]


def test_compiler_lowers_logical_bell_and_binds_real_measure_bit_operations() -> None:
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0)
    circuit.measure(1)
    metadata = _metadata(
        ("q0", "q1"),
        (MeasurementSlot("q0", 0), MeasurementSlot("q1", 1)),
        2,
    )

    artifact = CircuitCompiler().compile(
        TranslationBundle(circuit, metadata),
        options=CompilationOptions(
            mode=CompilationMode.ENHANCED,
            target_basis=_native_basis(),
        ),
    )

    assert sum(line.startswith("M ") for line in artifact.qcis.splitlines()) == 2
    assert [item.physical_qubit for item in artifact.measurements] == [0, 1]
    assert [item.classical_bit for item in artifact.measurements] == [0, 1]
    assert artifact.circuit.validate() is None
    assert all(
        _operation_name(operation) in set(_native_basis()) | {"MEASURE"}
        for operation in artifact.circuit.operations
    )
    assert any(step.name == "translate.target_basis" for step in artifact.steps)


def test_measure_bits_is_bound_after_qcis_expands_it() -> None:
    circuit = Circuit(2)
    circuit.x2p(0)
    circuit.measure_bits([0, 1])
    metadata = _metadata(
        ("q0", "q1"),
        (MeasurementSlot("q0", 1), MeasurementSlot("q1", 0)),
        2,
    )

    artifact = CircuitCompiler().compile(
        TranslationBundle(circuit, metadata),
        options=CompilationOptions(target_basis=("X2P",)),
    )

    assert [(item.physical_qubit, item.classical_bit) for item in artifact.measurements] == [
        (0, 1),
        (1, 0),
    ]
    assert [_operation_name(operation) for operation in artifact.circuit.operations][-2:] == [
        "MEASURE",
        "MEASURE",
    ]


def test_real_directives_do_not_require_direct_coupling_and_preserve_measurement_binding() -> None:
    config = Device.line("directive-line", 3)
    config.native_gates = _native_instructions()
    normalized = NormalizedDevice.from_backend(_NativeBackend(config))
    compiler = CircuitCompiler()

    assert not normalized.supports_coupling(0, 2, either_direction=True)

    barrier = Circuit(3)
    barrier.barrier([0, 2])
    barrier_artifact = compiler.compile(
        TranslationBundle(
            barrier,
            _metadata(("q0", "q1", "q2"), (), 0),
        ),
        device=normalized,
    )
    assert barrier_artifact.qcis == "B Q0 Q2"

    measured = Circuit(3)
    measured.measure_bits([0, 2])
    metadata = _metadata(
        ("q0", "q1", "q2"),
        (MeasurementSlot("q0", 0), MeasurementSlot("q2", 1)),
        2,
    )
    measurement_artifact = compiler.compile(
        TranslationBundle(measured, metadata),
        device=normalized,
    )

    compiled_qubits = [
        operation.qubits[0].index
        for operation in measurement_artifact.circuit.operations
        if _operation_name(operation) == "MEASURE"
    ]
    compiled_measurements = measurement_artifact.measurements
    actual_bindings = [(item.physical_qubit, item.classical_bit) for item in compiled_measurements]
    assert actual_bindings == [
        (compiled_qubits[0], 0),
        (compiled_qubits[1], 1),
    ]
    assert sum(line.startswith("M ") for line in measurement_artifact.qcis.splitlines()) == 1


def test_real_device_normalization_preserves_sparse_ids_invalid_qubits_and_instructions() -> None:
    topology = Topology([1, 8], [(1, 8, "CZ")])
    config = Device("sparse-native", [1, 8, 13], topology)
    config.invalid_qubits = [13]
    config.native_gates = _native_instructions()

    normalized = NormalizedDevice.from_backend(_NativeBackend(config))

    assert normalized.qubits == (1, 8, 13)
    assert normalized.usable_qubits == (1, 8)
    assert normalized.invalid_qubits == (13,)
    assert normalized.native_gates == _native_basis()
    assert normalized.supports_coupling(1, 8)
    assert normalized.supports_coupling(8, 1, either_direction=True)
    assert qubit_index(Qubit(8)) == 8


def test_compiler_routes_on_real_cqlib_device_and_respects_topology() -> None:
    config = Device.line("line-native", 3)
    config.native_gates = _native_instructions()
    normalized = NormalizedDevice.from_backend(_NativeBackend(config))

    circuit = Circuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.measure(0)
    circuit.measure(2)
    metadata = _metadata(
        ("q0", "q1", "q2"),
        (MeasurementSlot("q0", 0), MeasurementSlot("q2", 1)),
        2,
    )

    artifact = CircuitCompiler().compile(
        TranslationBundle(circuit, metadata),
        device=normalized,
        options=CompilationOptions(seed=101),
    )

    assert len(artifact.measurements) == 2
    assert any(step.name == "route.sabre" and not step.skipped for step in artifact.steps)
    for operation in artifact.circuit.operations:
        indices = tuple(qubit.index for qubit in operation.qubits)
        if len(indices) == 2:
            assert normalized.supports_coupling(*indices, either_direction=True)


def test_real_layout_and_resource_policy_map_measurement_to_physical_qubit() -> None:
    config = Device.line("layout-native", 3)
    config.native_gates = _native_instructions()
    normalized = NormalizedDevice.from_backend(_NativeBackend(config))
    circuit = Circuit(1)
    circuit.h(0)
    circuit.measure(0)
    metadata = _metadata(("q0",), (MeasurementSlot("q0", 0),), 1)

    artifact = CircuitCompiler().compile(
        TranslationBundle(circuit, metadata),
        device=normalized,
        options=CompilationOptions(
            initial_layout=Layout.from_pairs([(0, 2)], 3),
            resource_policy=ResourcePolicy(),
            seed=3,
        ),
    )

    assert artifact.measurements[0].physical_qubit == 2
    assert "M Q2" in artifact.qcis


def test_real_cqlib_compile_submit_wait_convert_closed_loop() -> None:
    config = Device.line("native-device", 2)
    config.native_gates = _native_instructions()
    native_result = ExecutionResult.from_counts(
        "native-cloud-task",
        [0],
        10,
        1,
        {"0": 7, "1": 3},
    )
    backend = _ExecutingNativeBackend(config, native_result)
    connector = TianyanConnector(_NativePlatform(backend))
    circuit = Circuit(1)
    circuit.x2p(0)
    circuit.measure(0)
    metadata = _metadata(("q0",), (MeasurementSlot("q0", 0),), 1)

    job = connector.compile_and_submit(
        [TranslationBundle(circuit, metadata)],
        run_options=RunOptions("native-device", shots=10),
    )
    results = job.wait(timeout=5, poll_interval=0.01)

    assert job.task_ids == ("native-cloud-task",)
    assert backend.submissions[0][1] == 10
    assert backend.submissions[0][0][0].endswith("M Q0")
    assert results[0].counts == {"0": 7, "1": 3}


def test_real_cqlib_execution_result_reconciles_calibrated_rounding() -> None:
    circuit = Circuit(2)
    circuit.measure(0)
    circuit.measure(1)
    metadata = _metadata(
        ("q0", "q1"),
        (MeasurementSlot("q0", 0), MeasurementSlot("q1", 1)),
        2,
    )
    artifact = CircuitCompiler().compile(TranslationBundle(circuit, metadata))
    raw = ExecutionResult.from_counts(
        "rounded-cloud-task",
        [0, 1],
        100,
        2,
        {"00": 50, "11": 49},
    )

    converted = ResultConverter().convert(raw, artifact)

    assert raw.shots == 100
    assert sum(raw.counts.values()) == 99
    assert raw.probabilities is not None
    assert converted.counts == {"00": 51, "11": 49}
    assert len(converted.samples) == 100


def test_validate_qcis_uses_real_parser_and_wraps_errors() -> None:
    compiler = CircuitCompiler()
    circuit = compiler.validate_qcis("X2P Q0\nCZ Q0 Q1\nM Q0\nM Q1")
    assert circuit.num_qubits == 2
    assert [_operation_name(operation) for operation in circuit.operations][-2:] == [
        "MEASURE",
        "MEASURE",
    ]

    with pytest.raises(AdapterCompileError, match="invalid QCIS"):
        compiler.validate_qcis("NOT_A_QCIS_GATE Q0")


def test_result_converter_accepts_real_cqlib_execution_result() -> None:
    circuit = Circuit([2, 0])
    circuit.measure(2)
    circuit.measure(0)
    metadata = _metadata(
        ("q2", "q0"),
        (MeasurementSlot("q2", 1), MeasurementSlot("q0", 0)),
        2,
    )
    artifact = CircuitCompiler().compile(TranslationBundle(circuit, metadata))
    result = ExecutionResult.from_counts(
        "native-result",
        [2, 0],
        100,
        2,
        {"10": 60, "01": 40},
    )

    converted = ResultConverter().convert(result, artifact)

    assert converted.task_id == "native-result"
    assert converted.counts == {"01": 60, "10": 40}
    assert converted.probabilities == pytest.approx({"01": 0.6, "10": 0.4})
    assert len(converted.samples) == 100


def test_compile_with_terminal_measurements_differs_only_by_output_phase() -> None:
    # cqlib's measurement-aware optimization (propagate_frames) drops
    # Z-diagonal phases on measured qubits, so compile guarantees
    # measurement-statistics equivalence rather than unitary equivalence:
    # the stripped compiled unitary V satisfies V = D @ U with D a diagonal
    # unitary. Assert exactly that, via M = V @ U.conj().T being diagonal
    # with unit-modulus entries.
    unitary_only = Circuit(2)
    unitary_only.ry(0, 0.37)
    unitary_only.cx(0, 1)
    unitary_only.ry(1, 0.61)

    measured = Circuit(2)
    measured.ry(0, 0.37)
    measured.cx(0, 1)
    measured.ry(1, 0.61)
    measured.measure(0)
    measured.measure(1)

    runtime = DefaultCqlibRuntime()
    compiled = runtime.compile(
        measured,
        mode=runtime.normal_mode(),
        target_basis=[name for name in _native_basis() if name != "GPHASE"],
        device=None,
        initial_layout=None,
        resource_policy=None,
        seed=None,
    )

    stripped = Circuit(2)
    for operation in compiled.circuit.operations:
        if _operation_name(operation) == "MEASURE":
            continue
        stripped.append(operation)

    source = _unitary_matrix(unitary_only, 2)
    product = _unitary_matrix(stripped, 2) @ source.conj().T
    off_diagonal = product - np.diag(np.diag(product))
    assert np.max(np.abs(off_diagonal)) < 1e-12
    assert np.allclose(np.abs(np.diag(product)), 1.0, atol=1e-12)
