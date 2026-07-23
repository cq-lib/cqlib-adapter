from __future__ import annotations

import os

import cirq
import pytest

from cqlib_adapter.cirq import TianyanSampler

pytestmark = [pytest.mark.cloud, pytest.mark.cirq]

RUN_CLOUD = os.getenv("CQLIB_RUN_CLOUD") == "1"
API_KEY = os.getenv("TIANYAN_API_KEY")
DEVICE_NAME = os.getenv("TIANYAN_DEVICE")


@pytest.mark.skipif(
    not RUN_CLOUD or not API_KEY or not DEVICE_NAME,
    reason=(
        "set CQLIB_RUN_CLOUD=1, TIANYAN_API_KEY and TIANYAN_DEVICE "
        "to authorize a real Tianyan Cirq submission"
    ),
)
def test_live_cirq_tianyan_closed_loop() -> None:
    shots = int(os.getenv("TIANYAN_TEST_SHOTS", "100"))
    timeout = float(os.getenv("TIANYAN_TEST_TIMEOUT", "300"))
    sampler = TianyanSampler.login(
        API_KEY,
        DEVICE_NAME,
        save_credentials=False,
        timeout=timeout,
    )
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.X(qubits[2]),
        cirq.measure(*qubits, key="state"),
    )

    result = sampler.run(circuit, repetitions=shots)
    print("task_ids:", sampler.last_task_ids)
    print("submitted QCIS:\n", sampler.last_qcis[0])
    print("histogram:", result.histogram(key="state"))

    assert sampler.last_task_ids
    assert result.repetitions == shots
    assert result.histogram(key="state").get(3, 0) / shots >= 0.5
    assert sampler.last_executions[0].result(timeout=timeout) is result
