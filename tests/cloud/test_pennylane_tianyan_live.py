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

import os

import pennylane as qml
import pytest

from cqlib_adapter.pennylane import TianyanDevice

pytestmark = [pytest.mark.cloud, pytest.mark.pennylane]

RUN_CLOUD = os.getenv("CQLIB_RUN_CLOUD") == "1"
API_KEY = os.getenv("TIANYAN_API_KEY")
DEVICE_NAME = os.getenv("TIANYAN_DEVICE")


@pytest.mark.skipif(
    not RUN_CLOUD or not API_KEY or not DEVICE_NAME,
    reason=(
        "set CQLIB_RUN_CLOUD=1, TIANYAN_API_KEY and TIANYAN_DEVICE "
        "to authorize a real Tianyan submission"
    ),
)
def test_live_pennylane_tianyan_closed_loop() -> None:
    shots = int(os.getenv("TIANYAN_TEST_SHOTS", "100"))
    timeout = float(os.getenv("TIANYAN_TEST_TIMEOUT", "300"))
    device = TianyanDevice.login(
        API_KEY,
        DEVICE_NAME,
        wires=3,
        save_credentials=False,
    )

    @qml.qnode(device, shots=shots)
    def circuit():
        qml.PauliX(1)
        qml.PauliX(2)
        return qml.counts(wires=(0, 1, 2))

    counts = circuit()
    print("task_ids:", device.last_task_ids)
    print("submitted QCIS:\n", device.last_qcis[0])
    print("counts:", counts)

    assert device.last_task_ids
    assert set(counts) <= {format(index, "03b") for index in range(8)}
    assert sum(counts.values()) == shots
    assert counts.get("011", 0) / shots >= 0.5
    assert device.last_executions[0].result(timeout=timeout) == counts
