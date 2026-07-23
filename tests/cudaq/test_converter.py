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
    qubits = cudaq.qvector(1)
    rx(theta, qubits[0])  # noqa: F821
    mz(qubits)  # noqa: F821


def operation_names(circuit: object) -> list[str]:
    return [
        operation.instruction.instruction.name  # type: ignore[attr-defined]
        for operation in circuit.operations  # type: ignore[attr-defined]
    ]


def test_explicit_mz_translates_through_openqasm2_to_real_cqlib() -> None:
    qasm = cudaq_to_openqasm(measured_011)
    bundle = cudaq_to_cqlib(measured_011)

    assert "OPENQASM 2" in qasm.upper()
    assert bundle.metadata.framework == "cudaq"
    assert bundle.metadata.qubits == ("q0", "q1", "q2")
    assert bundle.metadata.extras["auto_measure_all"] is False
    assert bundle.metadata.extras["measured_qubits"] == (0, 1, 2)
    assert len(bundle.metadata.measurements.slots) == 3
    assert operation_names(bundle.circuit)[-3:] == ["measure_bit"] * 3


def test_missing_mz_adds_full_final_measurement_before_cqlib_parse() -> None:
    bundle = cudaq_to_cqlib(unmeasured_bell)

    assert bundle.metadata.extras["auto_measure_all"] is True
    assert bundle.metadata.extras["measured_qubits"] == (0, 1)
    assert bundle.metadata.warnings == (
        "kernel had no explicit mz; the adapter added final measurement of every qubit",
    )
    assert "creg cqlib_mz[2];" in bundle.metadata.extras["openqasm2"]
    assert operation_names(bundle.circuit)[-2:] == ["measure_bit", "measure_bit"]


def test_parameterized_decorator_kernel_is_synthesized_before_openqasm() -> None:
    qasm = cudaq_to_openqasm(rotated, 0.25)
    bundle = cudaq_to_cqlib(rotated, 0.25)

    assert "rx(" in qasm.lower()
    assert bundle.metadata.circuit_name
    assert bundle.metadata.measurements.num_classical_bits == 1


def test_parameterized_builder_has_explicit_cudaq_015_error() -> None:
    builder, theta = cudaq.make_kernel(float)
    qubit = builder.qalloc()
    builder.rx(theta, qubit)
    builder.mz(qubit)

    with pytest.raises(AdapterConversionError, match="parameterized CUDA-Q PyKernel builders"):
        cudaq_to_openqasm(builder, 0.25)


def test_compile_cudaq_kernel_lowers_real_cqlib_to_native_qcis() -> None:
    artifact = compile_cudaq_kernel(
        unmeasured_bell,
        options=CompilationOptions(target_basis=NATIVE_BASIS, seed=19),
    )

    assert "CZ Q0 Q1" in artifact.qcis
    assert artifact.qcis.count("M Q") == 2
    assert [(item.physical_qubit, item.classical_bit) for item in artifact.measurements] == [
        (0, 0),
        (1, 1),
    ]


@pytest.mark.parametrize(
    ("qasm", "message"),
    [
        (
            """OPENQASM 2.0;
qreg first[1];
qreg second[1];
""",
            "exactly one qalloc/qvector",
        ),
        (
            """OPENQASM 2.0;
qreg q[1];
creg c[1];
if(c==1) x q[0];
""",
            "classical control",
        ),
        (
            """OPENQASM 2.0;
qreg q[1];
creg c[1];
measure q[0] -> c[0];
x q[0];
""",
            "operation follows mz",
        ),
    ],
)
def test_openqasm_boundaries_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
    qasm: str,
    message: str,
) -> None:
    monkeypatch.setattr(cudaq, "translate", lambda *_args, **_kwargs: qasm)

    with pytest.raises(AdapterConversionError, match=message):
        cudaq_to_cqlib(object())
