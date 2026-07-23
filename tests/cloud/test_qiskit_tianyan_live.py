from __future__ import annotations

import os

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("cqlib")
pytest.importorskip("cqlib_tianyan")
from qiskit import QuantumCircuit

from cqlib_adapter.qiskit import TianyanBackend

pytestmark = [pytest.mark.cloud, pytest.mark.qiskit]

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
def test_live_qiskit_tianyan_closed_loop() -> None:
    shots = int(os.getenv("TIANYAN_TEST_SHOTS", "100"))
    timeout = float(os.getenv("TIANYAN_TEST_TIMEOUT", "300"))
    backend = TianyanBackend.login(
        API_KEY,
        DEVICE_NAME,
        save_credentials=False,
    )
    circuit = QuantumCircuit(1, 1, name="qiskit-live-smoke")
    circuit.h(0)
    circuit.measure(0, 0)

    job = backend.run(circuit, shots=shots, calibration="auto")
    print("task_ids:", job.task_ids)
    print("submitted QCIS:\n", job.qcis[0])
    result = job.result(timeout=timeout)
    counts = result.get_counts()
    print("counts:", counts)

    assert job.task_ids
    assert set(counts) <= {"0", "1"}
    assert sum(counts.values()) == shots
