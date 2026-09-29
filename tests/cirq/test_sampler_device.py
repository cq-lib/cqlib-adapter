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
