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

cudaq = pytest.importorskip("cudaq")
pytest.importorskip("cqlib")

from cqlib_adapter.common import AdapterConversionError, CompilationOptions  # noqa: E402
from cqlib_adapter.cudaq.converter import (  # noqa: E402
    compile_cudaq_kernel,
    cudaq_to_cqlib,
    cudaq_to_openqasm,
)

pytestmark = pytest.mark.cudaq

NATIVE_BASIS = (
    "RZ",
    "X2P",
    "X2M",
    "Y2P",
    "Y2M",
    "XY2P",
    "XY2M",
    "CZ",
    "GPHASE",
)


@cudaq.kernel
def measured_011() -> None:
    qubits = cudaq.qvector(3)
    x(qubits[1])  # noqa: F821
    x(qubits[2])  # noqa: F821
    mz(qubits)  # noqa: F821


@cudaq.kernel
def unmeasured_bell() -> None:
    qubits = cudaq.qvector(2)
    h(qubits[0])  # noqa: F821
    x.ctrl(qubits[0], qubits[1])  # noqa: F821


@cudaq.kernel
def rotated(theta: float) -> None:
    qubit = cudaq.qubit()
    rx(theta, qubit)  # noqa: F821
    mz(qubit)  # noqa: F821


@cudaq.kernel
def grover_static_loops() -> None:
    qubits = cudaq.qvector(2)
    h(qubits)  # noqa: F821
    z.ctrl(qubits[0], qubits[1])  # noqa: F821
    h(qubits)  # noqa: F821
    x(qubits)  # noqa: F821
    z.ctrl(qubits[0], qubits[1])  # noqa: F821
    x(qubits)  # noqa: F821
    h(qubits)  # noqa: F821
    mz(qubits)  # noqa: F821


@cudaq.kernel
def mixed_basis_measurements() -> None:
    qubits = cudaq.qvector(3)
    h(qubits[0])  # noqa: F821
    h(qubits[1])  # noqa: F821
    s(qubits[1])  # noqa: F821
    mx(qubits[0])  # noqa: F821
    my(qubits[1])  # noqa: F821
    mz(qubits[2])  # noqa: F821


@cudaq.kernel
def operation_after_measurement() -> None:
    qubit = cudaq.qubit()
    mz(qubit)  # noqa: F821
    x(qubit)  # noqa: F821


def operation_names(circuit: object) -> list[str]:
    return [
        operation.instruction.instruction.name  # type: ignore[attr-defined]
        for operation in circuit.operations  # type: ignore[attr-defined]
    ]


def test_openqasm2_is_an_explicit_diagnostic_export_only() -> None:
    qasm = cudaq_to_openqasm(measured_011)

    assert "OPENQASM 2" in qasm.upper()


def test_direct_quake_path_never_calls_cudaq_translate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_translate(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("production conversion must not call cudaq.translate")

    monkeypatch.setattr(cudaq, "translate", fail_translate)
    bundle = cudaq_to_cqlib(measured_011)

    assert bundle.metadata.framework == "cudaq"
    assert bundle.metadata.qubits == ("q0", "q1", "q2")
    assert bundle.metadata.extras["source_ir"] == "quake"
    assert bundle.metadata.extras["auto_measure_all"] is False
    assert bundle.metadata.extras["measured_qubits"] == (0, 1, 2)
    assert len(bundle.metadata.measurements.slots) == 3
    assert operation_names(bundle.circuit)[-3:] == ["measure_bit"] * 3


def test_missing_measurement_adds_full_final_measurement() -> None:
    bundle = cudaq_to_cqlib(unmeasured_bell)

    assert bundle.metadata.extras["auto_measure_all"] is True
    assert bundle.metadata.extras["measured_qubits"] == (0, 1)
    assert bundle.metadata.warnings == (
        "kernel had no explicit measurement; the adapter added final mz for every qubit",
    )
    assert operation_names(bundle.circuit)[-2:] == ["measure_bit", "measure_bit"]


def test_parameterized_decorator_is_synthesized_to_concrete_quake() -> None:
    bundle = cudaq_to_cqlib(rotated, 0.25)

    assert operation_names(bundle.circuit) == ["RX", "measure_bit"]
    assert bundle.metadata.circuit_name
    assert bundle.metadata.measurements.num_classical_bits == 1


def test_parameterized_builder_and_scalar_arithmetic_translate_directly() -> None:
    builder, theta = cudaq.make_kernel(float)
    qubit = builder.qalloc()
    builder.rx(theta / 2.0, qubit)
    builder.mz(qubit)

    bundle = cudaq_to_cqlib(builder, 0.5)

    assert operation_names(bundle.circuit) == ["RX", "measure_bit"]
    assert bundle.metadata.extras["allocation_widths"] == (1,)
    with pytest.raises(AdapterConversionError, match="cannot be specialized to OpenQASM 2"):
        cudaq_to_openqasm(builder, 0.5)


def test_fixed_width_multiple_qallocs_share_one_stable_qubit_space() -> None:
    builder = cudaq.make_kernel()
    first = builder.qalloc(2)
    second = builder.qalloc()
    builder.x(first[1])
    builder.cx(first[1], second)
    builder.mz(first)
    builder.mz(second)

    bundle = cudaq_to_cqlib(builder)

    assert bundle.metadata.qubits == ("q0", "q1", "q2")
    assert bundle.metadata.extras["allocation_widths"] == (2, 1)
    assert bundle.metadata.extras["measured_qubits"] == (0, 1, 2)
    assert operation_names(bundle.circuit) == [
        "X",
        "CX",
        "measure_bit",
        "measure_bit",
        "measure_bit",
    ]


def test_static_cc_loops_are_evaluated_without_skipping_operations() -> None:
    bundle = cudaq_to_cqlib(grover_static_loops)
    names = operation_names(bundle.circuit)

    assert names == [
        "H",
        "H",
        "CZ",
        "H",
        "H",
        "X",
        "X",
        "CZ",
        "X",
        "X",
        "H",
        "H",
        "measure_bit",
        "measure_bit",
    ]


def test_cudaq_mx_my_mz_are_lowered_to_basis_rotations_and_measurements() -> None:
    bundle = cudaq_to_cqlib(mixed_basis_measurements)

    assert bundle.metadata.extras["measurement_bases"] == ("X", "Y", "Z")
    assert bundle.metadata.extras["measured_qubits"] == (0, 1, 2)
    assert operation_names(bundle.circuit)[-6:] == [
        "H",
        "SDG",
        "H",
        "measure_bit",
        "measure_bit",
        "measure_bit",
    ]


def test_compile_cudaq_kernel_lowers_real_cqlib_to_native_qcis() -> None:
    artifact = compile_cudaq_kernel(
        unmeasured_bell,
        options=CompilationOptions(target_basis=NATIVE_BASIS),
    )

    assert "CZ Q0 Q1" in artifact.qcis
    assert sum(line.startswith("M ") for line in artifact.qcis.splitlines()) == 2
    assert [(item.physical_qubit, item.classical_bit) for item in artifact.measurements] == [
        (0, 0),
        (1, 1),
    ]


def test_dynamic_qalloc_is_rejected_instead_of_guessed_or_exported() -> None:
    builder, width = cudaq.make_kernel(int)
    builder.qalloc(width)

    with pytest.raises(AdapterConversionError, match=r"dynamic quake\.alloca"):
        cudaq_to_cqlib(builder, 3)


def test_gate_after_measurement_is_rejected() -> None:
    with pytest.raises(AdapterConversionError, match="follows a measurement"):
        cudaq_to_cqlib(operation_after_measurement)


def test_unknown_quake_operation_is_never_silently_skipped() -> None:
    builder, theta = cudaq.make_kernel(float)
    qubits = builder.qalloc(2)
    builder.exp_pauli(theta, qubits, "XX")

    with pytest.raises(AdapterConversionError, match="no operation was skipped"):
        cudaq_to_cqlib(builder, 0.25)


def test_builder_argument_count_is_validated() -> None:
    builder, theta = cudaq.make_kernel(float)
    qubit = builder.qalloc()
    builder.rx(theta, qubit)

    with pytest.raises(AdapterConversionError, match="expects 1 arguments, received 0"):
        cudaq_to_cqlib(builder)
