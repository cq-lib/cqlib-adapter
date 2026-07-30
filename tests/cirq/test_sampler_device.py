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

from math import pi

import cirq
import numpy as np
import pytest
import sympy
from cqlib.device import Layout

from cqlib_adapter.cirq import CqlibSimulatorSampler
from cqlib_adapter.cirq.testing import ResultSpec, make_cirq_sampler
from cqlib_adapter.common import AdapterDeviceError, JobState

pytestmark = pytest.mark.cirq


def test_mock_sampler_returns_standard_result_keys_and_task_metadata() -> None:
    sampler, cloud = make_cirq_sampler(
        [ResultSpec({"110": 10}, (0, 1, 2), status_ready=True)],
        size=3,
    )
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.X(qubits[2]),
        cirq.measure(qubits[0], key="left"),
        cirq.measure(qubits[1], qubits[2], key="pair"),
    )

    result = sampler.run(circuit, repetitions=10)

    assert isinstance(result, cirq.ResultDict)
    assert result.histogram(key="left") == {0: 10}
    assert result.histogram(key="pair") == {3: 10}
    np.testing.assert_array_equal(result.measurements["pair"], [[True, True]] * 10)
    assert cloud.calls[0][2] == 10
    assert sampler.last_task_ids == ("mock-cirq-task-1",)
    assert sampler.last_executions[0].status().state is JobState.DONE
    assert "M Q0" in sampler.last_qcis[0]


def test_submit_returns_queryable_task_and_probabilities_before_result() -> None:
    sampler, cloud = make_cirq_sampler(
        [ResultSpec({"110": 10}, (0, 1, 2))],
        size=3,
    )
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(qubits[0], key="left"),
        cirq.measure(qubits[1], qubits[2], key="pair"),
    )

    execution = sampler.submit(circuit, repetitions=10)

    assert execution.task_id == "mock-cirq-task-1"
    assert execution.status().state is JobState.SUBMITTED
    assert sampler.last_executions == (execution,)
    assert cloud.handles[0].wait_calls == []
    assert execution.probabilities("left") == {0: 1.0}
    assert execution.probabilities("pair") == {3: 1.0}
    assert execution.result().histogram(key="pair") == {3: 10}
    assert len(cloud.handles[0].wait_calls) == 1


def test_sampler_availability_refreshes_cloud_state() -> None:
    sampler, cloud = make_cirq_sampler([], size=1)
    cloud.status = "calibration"
    cloud._available = False

    assert not sampler.is_available()
    assert sampler.device.status.value == "calibration"


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
def test_sampler_rejects_invalid_wait_configuration(
    keyword: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        make_cirq_sampler([], size=1, **{keyword: value})


def test_run_sweep_resolves_parameters_and_preserves_resolver_order() -> None:
    sampler, _cloud = make_cirq_sampler(
        [
            ResultSpec({"0": 4}, (0,)),
            ResultSpec({"1": 4}, (0,)),
        ],
        size=1,
    )
    qubit = cirq.LineQubit(0)
    theta = sympy.Symbol("theta")
    circuit = cirq.Circuit(cirq.rx(theta)(qubit), cirq.measure(qubit, key="m"))

    results = sampler.run_sweep(circuit, params=[{"theta": 0.0}, {"theta": pi}], repetitions=4)

    assert [result.params.value_of("theta") for result in results] == [0.0, pi]
    assert [result.histogram(key="m") for result in results] == [{0: 4}, {1: 4}]
    assert sampler.last_task_ids == ("mock-cirq-task-1", "mock-cirq-task-2")


def test_inherited_sample_dataframe_uses_big_endian_measurement_integer() -> None:
    sampler, _cloud = make_cirq_sampler([ResultSpec({"110": 3}, (0, 1, 2))], size=3)
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.X(qubits[1]), cirq.X(qubits[2]), cirq.measure(*qubits, key="m"))

    frame = sampler.sample(circuit, repetitions=3)

    assert frame["m"].tolist() == [3, 3, 3]


def test_device_metadata_exposes_topology_and_validation() -> None:
    sampler, _cloud = make_cirq_sampler([], size=3)
    q0, q1, q2 = cirq.LineQubit.range(3)

    assert sampler.device.num_qubits == 3
    assert sampler.device.available
    assert sampler.device.metadata.qubit_set == frozenset({q0, q1, q2})
    assert sampler.device.metadata.nx_graph.has_edge(q0, q1)
    sampler.device.validate_operation(cirq.CZ(q0, q1))
    with pytest.raises(ValueError, match="not coupled"):
        sampler.device.validate_operation(cirq.CZ(q0, q2))
    with pytest.raises(ValueError, match="outside"):
        sampler.device.validate_operation(cirq.X(cirq.LineQubit(9)))


def test_unavailable_device_and_invalid_repetitions_fail_before_submission() -> None:
    sampler, cloud = make_cirq_sampler(
        [ResultSpec({"0": 2}, (0,))],
        size=1,
        available=False,
    )
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(qubit), cirq.measure(qubit, key="m"))

    with pytest.raises(AdapterDeviceError, match="not available"):
        sampler.run(circuit, repetitions=2)
    assert cloud.calls == []
    with pytest.raises(ValueError, match="positive integer"):
        sampler.run(circuit, repetitions=0)


def test_local_cqlib_sampler_preserves_011_measurement_order() -> None:
    sampler = CqlibSimulatorSampler(3)
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.X(qubits[2]),
        cirq.measure(*qubits, key="state"),
    )

    result = sampler.run(circuit, repetitions=32)

    assert result.histogram(key="state") == {3: 32}
    np.testing.assert_array_equal(result.measurements["state"], [[False, True, True]] * 32)
    assert sampler.simulator_calls


def test_local_cqlib_sampler_grover_finds_marked_state() -> None:
    sampler = CqlibSimulatorSampler(2)
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.H(q1),
        cirq.CZ(q0, q1),
        cirq.H(q0),
        cirq.H(q1),
        cirq.X(q0),
        cirq.X(q1),
        cirq.CZ(q0, q1),
        cirq.X(q0),
        cirq.X(q1),
        cirq.H(q0),
        cirq.H(q1),
        cirq.measure(q0, q1, key="result"),
    )

    result = sampler.run(circuit, repetitions=64)

    assert result.histogram(key="result") == {3: 64}


def test_local_cqlib_sampler_supports_explicit_x_and_y_basis_measurement() -> None:
    sampler = CqlibSimulatorSampler(2)
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),  # Prepare |+>.
        cirq.H(q1),
        cirq.S(q1),  # Prepare |+i>.
        cirq.H(q0),  # X-basis measurement rotation.
        cirq.S(q1) ** -1,
        cirq.H(q1),  # Y-basis measurement rotation.
        cirq.measure(q0, q1, key="basis"),
    )

    result = sampler.run(circuit, repetitions=32)

    assert result.histogram(key="basis") == {0: 32}


def test_local_cqlib_run_statevector_matches_cirq_amplitudes() -> None:
    qubits = cirq.LineQubit.range(2)
    unitary = cirq.Circuit(
        cirq.ry(0.37)(qubits[0]),
        cirq.rz(-0.29)(qubits[1]),
        cirq.CNOT(*qubits),
        cirq.ry(0.61)(qubits[1]),
    )
    measured = unitary + cirq.Circuit(cirq.measure(*qubits, key="result"))
    sampler = CqlibSimulatorSampler(2)

    actual = sampler.run_statevector(
        measured,
        qubit_order=qubits,
    )
    reference = cirq.final_state_vector(
        unitary,
        qubit_order=qubits,
        dtype=np.complex128,
    )
    fidelity = abs(np.vdot(reference, np.asarray(actual.data))) ** 2

    assert fidelity == pytest.approx(1.0, abs=1e-12)
    assert actual.qubit_order == tuple(qubits)
    assert actual.physical_qubits == (0, 1)
    assert "CZ Q0 Q1" in actual.qcis
    assert "CX " not in actual.qcis
    assert not any(line.startswith("M ") for line in actual.qcis.splitlines())
    assert sampler.simulator_calls == ()


def test_local_cqlib_run_statevector_restores_reversed_layout() -> None:
    qubits = cirq.LineQubit.range(3)
    layout = Layout.from_pairs([(0, 2), (1, 1), (2, 0)], physical_count=3)
    circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.H(qubits[1]),
        cirq.ry(0.37)(qubits[2]),
    )
    sampler = CqlibSimulatorSampler(3, initial_layout=layout, seed=47)

    actual = sampler.run_statevector(circuit, qubit_order=qubits)
    reference = cirq.final_state_vector(
        circuit,
        qubit_order=qubits,
        dtype=np.complex128,
    )

    assert abs(np.vdot(reference, np.asarray(actual.data))) ** 2 == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert actual.physical_qubits == (2, 1, 0)


def test_local_cqlib_run_statevector_restores_layout_after_routing() -> None:
    qubits = cirq.LineQubit.range(3)
    layout = Layout.from_pairs([(0, 2), (1, 1), (2, 0)], physical_count=3)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.ry(0.37)(qubits[2]),
        cirq.CNOT(qubits[0], qubits[2]),
        cirq.rz(-0.29)(qubits[1]),
    )
    sampler = CqlibSimulatorSampler(3, initial_layout=layout, seed=53)

    actual = sampler.run_statevector(circuit, qubit_order=qubits)
    reference = cirq.final_state_vector(
        circuit,
        qubit_order=qubits,
        dtype=np.complex128,
    )

    assert abs(np.vdot(reference, np.asarray(actual.data))) ** 2 == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert set(actual.physical_qubits) == {0, 1, 2}
    assert any(step.name == "route.sabre" and not step.skipped for step in actual.artifact.steps)


def test_local_cqlib_run_statevector_rejects_mid_measurement() -> None:
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.H(qubit),
        cirq.measure(qubit, key="middle"),
        cirq.H(qubit),
    )

    with pytest.raises(
        Exception,
        match="only terminal measurements",
    ):
        CqlibSimulatorSampler(1).run_statevector(circuit)


def test_local_cqlib_run_statevector_rejects_width_mismatch() -> None:
    qubit = cirq.LineQubit(0)

    with pytest.raises(ValueError, match="use every simulator qubit"):
        CqlibSimulatorSampler(2).run_statevector(cirq.Circuit(cirq.H(qubit)))


def test_local_cqlib_sampler_supports_measurement_only_zero_state() -> None:
    sampler = CqlibSimulatorSampler(3)
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(*qubits, key="zero"))

    result = sampler.run(circuit, repetitions=12)

    assert result.histogram(key="zero") == {0: 12}
    np.testing.assert_array_equal(result.measurements["zero"], [[False] * 3] * 12)
