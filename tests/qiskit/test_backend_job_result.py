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

pytest.importorskip("qiskit")
pytest.importorskip("cqlib")
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.providers import JobStatus
from qiskit.providers.exceptions import JobError
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.result import Result

from cqlib_adapter.common import AdapterConversionError, AdapterDeviceError
from cqlib_adapter.qiskit import CqlibSimulatorBackend, TianyanJob, TianyanSampler
from cqlib_adapter.qiskit.testing import ResultSpec, make_qiskit_backend

pytestmark = pytest.mark.qiskit


def bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2, name="bell")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def test_backend_exposes_qiskit_target_options_and_device_status() -> None:
    backend, _ = make_qiskit_backend([], size=2)

    assert backend.name == "mock-qpu"
    assert backend.num_qubits == 2
    assert backend.options.shots == 1024
    assert backend.max_circuits == 50


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("timeout", float("nan"), ValueError),
        ("poll_interval", float("inf"), ValueError),
        ("timeout", float("-inf"), ValueError),
        ("poll_interval", float("-inf"), ValueError),
        ("timeout", True, TypeError),
        ("poll_interval", "5", TypeError),
    ],
)
def test_backend_rejects_invalid_wait_configuration_before_submission(
    keyword: str,
    value: object,
    error: type[Exception],
) -> None:
    backend, cloud = make_qiskit_backend([], size=2)

    with pytest.raises(error):
        backend.run(bell_circuit(), **{keyword: value})

    assert cloud.calls == []
    assert backend.is_available()
    assert backend.status().operational
    assert backend.status().status_msg == "running"
    assert "cz" in backend.operation_names
    assert TianyanSampler(backend).backend is backend


def test_backend_status_refreshes_cloud_state() -> None:
    backend, cloud = make_qiskit_backend([], size=1)
    cloud.status = "under_maintenance"
    cloud._available = False

    status = backend.status()

    assert not status.operational
    assert status.status_msg == "under_maintenance"
    assert backend.device_status.value == "under_maintenance"


def test_backend_mock_closed_loop_returns_standard_qiskit_result() -> None:
    backend, cloud = make_qiskit_backend(
        [ResultSpec({"00": 70, "11": 30}, (0, 1))],
        size=2,
    )

    job = backend.run(bell_circuit(), shots=100, poll_interval=0.01)

    assert isinstance(job, TianyanJob)
    assert job.job_id() == "mock-task-1"
    assert job.task_ids == ("mock-task-1",)
    assert job.status() is JobStatus.QUEUED
    assert "CZ Q0 Q1" in job.qcis[0]
    assert sum(line.startswith("M ") for line in job.qcis[0].splitlines()) == 2
    assert cloud.calls == [("auto", [job.qcis[0]], 100)]

    result = job.result(timeout=2, poll_interval=0.01)

    assert isinstance(result, Result)
    assert result.get_counts() == {"00": 70, "11": 30}
    assert len(result.get_memory()) == 100
    assert result.data()["probabilities"] == pytest.approx({"00": 0.7, "11": 0.3})
    assert result.results[0].header["task_id"] == "mock-task-1"
    assert job.status() is JobStatus.DONE
    assert job.result() is result


def test_backend_sampler_v2_executes_the_same_mock_cloud_loop() -> None:
    backend, _ = make_qiskit_backend(
        [ResultSpec({"0": 7, "1": 3}, (0,))],
        size=1,
    )
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)

    primitive_result = TianyanSampler(backend).run([circuit], shots=10).result()

    assert primitive_result[0].data.c.get_counts() == {"0": 7, "1": 3}


def test_batch_results_preserve_submitted_circuit_order() -> None:
    backend, cloud = make_qiskit_backend(
        [
            ResultSpec({"0": 8, "1": 2}, (0,)),
            ResultSpec({"0": 3, "1": 7}, (0,)),
        ],
        size=1,
    )
    zero = QuantumCircuit(1, 1, name="zero")
    zero.measure(0, 0)
    one = QuantumCircuit(1, 1, name="one")
    one.x(0)
    one.measure(0, 0)

    job = backend.run([zero, one], shots=10)
    result = job.result(timeout=2)

    assert job.task_ids == ("mock-task-1", "mock-task-2")
    assert result.get_counts() == [{"0": 8, "1": 2}, {"0": 3, "1": 7}]
    assert [entry.header["name"] for entry in result.results] == ["zero", "one"]
    assert len(cloud.calls) == 2


def test_calibration_mode_selects_cqlib_tianyan_run_raw() -> None:
    backend, cloud = make_qiskit_backend(
        [ResultSpec({"0": 10}, (0,))],
        size=1,
    )
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    backend.run(circuit, shots=10, calibration="disabled")

    assert cloud.calls[0][0] == "disabled"


def test_multiple_classical_registers_use_qiskit_display_order() -> None:
    backend, _ = make_qiskit_backend(
        [ResultSpec({"10": 10}, (0, 1))],
        size=2,
    )
    qreg = QuantumRegister(2, "q")
    left = ClassicalRegister(1, "left")
    right = ClassicalRegister(1, "right")
    circuit = QuantumCircuit(qreg, left, right, name="register-order")
    circuit.measure(qreg[0], right[0])
    circuit.measure(qreg[1], left[0])

    result = backend.run(circuit, shots=10).result(timeout=2)

    assert result.get_counts() == {"0 1": 10}
    assert result.results[0].header["creg_sizes"] == [["left", 1], ["right", 1]]


def test_job_submit_and_cancel_follow_immediate_submission_contract() -> None:
    backend, _ = make_qiskit_backend(
        [ResultSpec({"0": 1}, (0,), status_ready=True)],
        size=1,
    )
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    job = backend.run(circuit, shots=1)

    assert job.status() is JobStatus.DONE
    assert job.cancel() is False
    with pytest.raises(JobError, match="already submitted"):
        job.submit()


def test_backend_rejects_invalid_inputs_and_missing_measurements() -> None:
    backend, _ = make_qiskit_backend([], size=2, max_circuits=1)
    no_measure = QuantumCircuit(1)
    no_measure.x(0)

    with pytest.raises(TypeError, match="run_input"):
        backend.run("not-a-circuit")
    with pytest.raises(ValueError, match="at most 1"):
        backend.run([bell_circuit(), bell_circuit()])
    with pytest.raises(AdapterConversionError, match="no measurements"):
        backend.run(no_measure)
    with pytest.raises(TypeError, match="unsupported Tianyan backend run options"):
        backend.run(bell_circuit(), unknown_option=True)


def test_unavailable_device_is_rejected_before_cloud_submission() -> None:
    backend, cloud = make_qiskit_backend(
        [ResultSpec({"00": 1}, (0, 1))],
        size=2,
        available=False,
    )

    with pytest.raises(AdapterDeviceError, match="not available"):
        backend.run(bell_circuit(), shots=1)
    assert cloud.calls == []


def test_local_cqlib_simulator_backend_executes_and_maps_partial_measurement() -> None:
    backend = CqlibSimulatorBackend(2)
    circuit = QuantumCircuit(2, 1, name="local-dj")
    circuit.x(1)
    circuit.h([0, 1])
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure(0, 0)

    job = backend.run(circuit, shots=64, seed=23)
    result = job.result(timeout=2, poll_interval=0.01)

    assert result.get_counts() == {"1": 64}
    assert result.results[0].header["name"] == "local-dj"
    assert job.qcis[0].endswith("M Q0")
    assert backend.simulator_calls[0][2] == 64


def test_local_cqlib_simulator_supports_explicit_x_and_y_basis_measurement() -> None:
    backend = CqlibSimulatorBackend(2)
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)  # Prepare |+>.
    circuit.h(1)
    circuit.s(1)  # Prepare |+i>.
    circuit.h(0)  # X-basis measurement rotation.
    circuit.sdg(1)
    circuit.h(1)  # Y-basis measurement rotation.
    circuit.measure([0, 1], [0, 1])

    result = backend.run(circuit, shots=32, seed=29).result()

    assert result.get_counts() == {"00": 32}


def test_local_cqlib_run_statevector_preserves_amplitudes_and_phase() -> None:
    backend = CqlibSimulatorBackend(2)
    circuit = QuantumCircuit(2, 2)
    circuit.ry(0.37, 0)
    circuit.rz(-0.29, 1)
    circuit.cx(0, 1)
    circuit.ry(0.61, 1)
    circuit.measure([0, 1], [0, 1])

    actual = backend.run_statevector(circuit, seed=17)
    reference = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))

    assert state_fidelity(reference, Statevector(list(actual.data))) == pytest.approx(
        1.0, abs=1e-12
    )
    assert actual.num_qubits == 2
    assert actual.physical_qubits == (0, 1)
    assert "CZ Q0 Q1" in actual.qcis
    assert "CX " not in actual.qcis
    assert not any(line.startswith("M ") for line in actual.qcis.splitlines())
    assert backend.simulator_calls == ()


@pytest.mark.parametrize("operation", ["measure", "reset"])
def test_local_cqlib_run_statevector_rejects_nonunitary_middle_operation(
    operation: str,
) -> None:
    backend = CqlibSimulatorBackend(1)
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    if operation == "measure":
        circuit.measure(0, 0)
    else:
        circuit.reset(0)
    circuit.h(0)

    with pytest.raises(
        AdapterConversionError,
        match="mid-circuit measurement or reset",
    ):
        backend.run_statevector(circuit)


def test_local_cqlib_run_statevector_rejects_width_mismatch() -> None:
    backend = CqlibSimulatorBackend(2)

    with pytest.raises(ValueError, match="width must equal"):
        backend.run_statevector(QuantumCircuit(1))


def test_local_cqlib_simulator_backend_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="num_qubits must be positive"):
        CqlibSimulatorBackend(0)
