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

import numpy as np
import pennylane as qml
import pytest

pytest.importorskip("cqlib")
from cqlib.device import Layout
from pennylane.tape import QuantumScript

from cqlib_adapter.common import AdapterConversionError, AdapterDeviceError, JobState
from cqlib_adapter.pennylane import CqlibSimulatorDevice
from cqlib_adapter.pennylane.testing import ResultSpec, make_pennylane_device

pytestmark = pytest.mark.pennylane


def test_qnode_mock_closed_loop_returns_counts_probs_sample_and_task_metadata() -> None:
    device, cloud = make_pennylane_device(
        [ResultSpec({"110": 10}, (0, 1, 2), status_ready=True)],
        wires=3,
    )

    @qml.qnode(device, shots=10)
    def circuit() -> tuple[dict[str, int], np.ndarray, np.ndarray]:
        qml.PauliX(1)
        qml.PauliX(2)
        return (
            qml.counts(wires=(0, 1, 2)),
            qml.probs(wires=(0, 1, 2)),
            qml.sample(wires=(0, 1, 2)),
        )

    counts, probabilities, samples = circuit()

    assert counts == {"011": 10}
    np.testing.assert_array_equal(probabilities, [0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(samples, [[0, 1, 1]] * 10)
    assert cloud.calls[0][0] == "auto"
    assert cloud.calls[0][2] == 10
    assert device.last_task_ids == ("mock-pennylane-task-1",)
    assert device.last_executions[0].status().state is JobState.DONE
    assert "M Q0" in device.last_qcis[0]


def test_direct_batch_execute_preserves_circuit_order() -> None:
    device, _cloud = make_pennylane_device(
        [
            ResultSpec({"0": 4}, (0,)),
            ResultSpec({"1": 4}, (0,)),
        ],
        wires=1,
        size=1,
    )
    first = QuantumScript([qml.Identity(0)], [qml.counts(wires=0)], shots=4)
    second = QuantumScript([qml.PauliX(0)], [qml.counts(wires=0)], shots=4)

    results = device.execute((first, second))

    assert results == ({"0": 4}, {"1": 4})
    assert device.last_task_ids == ("mock-pennylane-task-1", "mock-pennylane-task-2")


def test_submit_returns_queryable_task_before_waiting() -> None:
    device, cloud = make_pennylane_device(
        [ResultSpec({"0": 3}, (0,))],
        wires=1,
        size=1,
    )
    tape = QuantumScript([qml.Identity(0)], [qml.counts(wires=0)], shots=3)

    execution = device.submit(tape)

    assert execution.task_id == "mock-pennylane-task-1"
    assert execution.status().state is JobState.SUBMITTED
    assert device.last_executions == (execution,)
    assert cloud.handles[0].wait_calls == []
    assert execution.result(timeout=2, poll_interval=0.01) == {"0": 3}
    assert len(cloud.handles[0].wait_calls) == 1


def test_device_availability_refreshes_cloud_state() -> None:
    device, cloud = make_pennylane_device([], wires=1, size=1)
    cloud.status = "offline"
    cloud._available = False

    assert not device.is_available()
    assert device.device_status.value == "offline"


def test_finite_shots_and_non_partitioned_shots_are_required() -> None:
    device, _cloud = make_pennylane_device([], wires=1, size=1)
    analytic = QuantumScript([qml.PauliX(0)], [qml.probs(wires=0)])
    partitioned = QuantumScript(
        [qml.PauliX(0)],
        [qml.counts(wires=0)],
        shots=(2, 3),
    )

    with pytest.raises(AdapterConversionError, match="finite shots"):
        device.execute(analytic)
    with pytest.raises(AdapterConversionError, match="shot vectors"):
        device.execute(partitioned)


def test_unavailable_cloud_device_is_rejected_before_submission() -> None:
    device, cloud = make_pennylane_device(
        [ResultSpec({"0": 2}, (0,))],
        wires=1,
        size=1,
        available=False,
    )
    tape = QuantumScript([qml.Identity(0)], [qml.counts(wires=0)], shots=2)

    with pytest.raises(AdapterDeviceError, match="not available"):
        device.execute(tape)
    assert cloud.calls == []


def test_named_wires_are_preserved_at_the_pennylane_boundary() -> None:
    device = CqlibSimulatorDevice(wires=("first", "second", "third"))

    @qml.qnode(device, shots=20)
    def circuit() -> dict[str, int]:
        qml.PauliX("second")
        qml.PauliX("third")
        return qml.counts(wires=("first", "second", "third"))

    assert circuit() == {"011": 20}


def test_local_cqlib_simulator_grover_qnode_finds_marked_state() -> None:
    device = CqlibSimulatorDevice(wires=2)

    @qml.qnode(device, shots=64)
    def grover() -> dict[str, int]:
        for wire in (0, 1):
            qml.Hadamard(wire)
        qml.CZ((0, 1))
        for wire in (0, 1):
            qml.Hadamard(wire)
            qml.PauliX(wire)
        qml.CZ((0, 1))
        for wire in (0, 1):
            qml.PauliX(wire)
            qml.Hadamard(wire)
        return qml.counts(wires=(0, 1))

    assert grover() == {"11": 64}
    assert device.simulator_calls[0][2] == 64
    assert "CZ Q0 Q1" in device.last_qcis[0]


def test_local_cqlib_simulator_supports_pauli_basis_measurements() -> None:
    device = CqlibSimulatorDevice(wires=1)

    @qml.qnode(device, shots=32)
    def measure_x() -> tuple[dict[float, int], np.ndarray, np.ndarray, float, float]:
        qml.Hadamard(0)
        return (
            qml.counts(qml.X(0), all_outcomes=True),
            qml.probs(op=qml.X(0)),
            qml.sample(qml.X(0)),
            qml.expval(qml.X(0)),
            qml.var(qml.X(0)),
        )

    @qml.qnode(device, shots=32)
    def measure_y() -> float:
        qml.Hadamard(0)
        qml.S(0)
        return qml.expval(qml.Y(0))

    counts, probabilities, samples, expectation, variance = measure_x()
    assert counts == {1.0: 32, -1.0: 0}
    np.testing.assert_array_equal(probabilities, [1.0, 0.0])
    np.testing.assert_array_equal(samples, [1.0] * 32)
    assert expectation == pytest.approx(1.0)
    assert variance == pytest.approx(0.0)
    assert measure_y() == pytest.approx(1.0)


def test_local_cqlib_run_statevector_matches_pennylane_amplitudes() -> None:
    operations = [
        qml.RY(0.37, wires=0),
        qml.RZ(-0.29, wires=1),
        qml.CNOT((0, 1)),
        qml.RY(0.61, wires=1),
    ]
    tape = QuantumScript(operations, [qml.state()])
    device = CqlibSimulatorDevice(wires=2)

    actual = device.run_statevector(tape)
    reference = qml.matrix(
        QuantumScript(operations, []),
        wire_order=(0, 1),
    ) @ np.array([1, 0, 0, 0], dtype=complex)
    fidelity = abs(np.vdot(reference, np.asarray(actual.data))) ** 2

    assert fidelity == pytest.approx(1.0, abs=1e-12)
    assert actual.wire_order == (0, 1)
    assert actual.physical_qubits == (0, 1)
    assert "CZ Q0 Q1" in actual.qcis
    assert "CX " not in actual.qcis
    assert not any(line.startswith("M ") for line in actual.qcis.splitlines())
    assert device.simulator_calls == ()


def test_local_cqlib_run_statevector_preserves_named_wire_order() -> None:
    device = CqlibSimulatorDevice(wires=("first", "second", "third"))
    tape = QuantumScript(
        [
            qml.PauliX("second"),
            qml.PauliX("third"),
        ],
        [],
    )

    actual = np.asarray(device.run_statevector(tape).data)
    expected = np.zeros(8, dtype=complex)
    expected[3] = 1.0

    assert abs(np.vdot(expected, actual)) ** 2 == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_local_cqlib_run_statevector_supports_empty_zero_state() -> None:
    actual = CqlibSimulatorDevice(wires=2).run_statevector(QuantumScript([], [qml.state()]))

    np.testing.assert_array_equal(actual.data, [1, 0, 0, 0])


def test_local_cqlib_run_statevector_restores_reversed_layout() -> None:
    layout = Layout.from_pairs([(0, 2), (1, 1), (2, 0)], physical_count=3)
    operations = [
        qml.PauliX(0),
        qml.Hadamard(1),
        qml.RY(0.37, wires=2),
    ]
    tape = QuantumScript(operations, [qml.state()])
    device = CqlibSimulatorDevice(wires=3, initial_layout=layout, seed=47)

    actual = device.run_statevector(tape)
    reference = qml.matrix(
        QuantumScript(operations, []),
        wire_order=(0, 1, 2),
    ) @ np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)

    assert abs(np.vdot(reference, np.asarray(actual.data))) ** 2 == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert actual.physical_qubits == (2, 1, 0)


def test_local_cqlib_run_statevector_restores_layout_after_routing() -> None:
    layout = Layout.from_pairs([(0, 2), (1, 1), (2, 0)], physical_count=3)
    operations = [
        qml.Hadamard(0),
        qml.RY(0.37, wires=2),
        qml.CNOT((0, 2)),
        qml.RZ(-0.29, wires=1),
    ]
    tape = QuantumScript(operations, [qml.state()])
    device = CqlibSimulatorDevice(wires=3, initial_layout=layout, seed=53)

    actual = device.run_statevector(tape)
    reference = qml.matrix(
        QuantumScript(operations, []),
        wire_order=(0, 1, 2),
    ) @ np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)

    assert abs(np.vdot(reference, np.asarray(actual.data))) ** 2 == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert set(actual.physical_qubits) == {0, 1, 2}
    assert any(step.name == "route.sabre" and not step.skipped for step in actual.artifact.steps)


def test_qnode_preprocess_decomposes_rot_before_cqlib_translation() -> None:
    device = CqlibSimulatorDevice(wires=1)

    @qml.qnode(device, shots=16)
    def circuit() -> dict[str, int]:
        qml.Rot(0.1, 0.2, 0.3, wires=0)
        return qml.counts(wires=0)

    result = circuit()
    assert sum(result.values()) == 16
    assert "M Q0" in device.last_qcis[0]


def test_device_exposes_cloud_metadata_and_validates_wire_capacity() -> None:
    device, _cloud = make_pennylane_device([], wires=2, size=3)
    assert device.name == "cqlib.tianyan"
    assert device.device.num_qubits == 3
    assert device.is_available()
    assert "CNOT" in device.operations
    with pytest.raises(ValueError, match="only 3"):
        make_pennylane_device([], wires=4, size=3)
