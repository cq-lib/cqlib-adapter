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

from cqlib_adapter.common import AdapterDeviceError, JobState  # noqa: E402
from cqlib_adapter.cudaq.testing import ResultSpec, make_cudaq_executor  # noqa: E402

pytestmark = pytest.mark.cudaq


@cudaq.kernel
def state_110() -> None:
    qubits = cudaq.qvector(3)
    x(qubits[0])  # noqa: F821
    x(qubits[1])  # noqa: F821
    mz(qubits)  # noqa: F821


@cudaq.kernel
def zero() -> None:
    qubit = cudaq.qubit()
    mz(qubit)  # noqa: F821


def test_mock_sync_sample_returns_cudaq_familiar_counts_and_metadata() -> None:
    executor, cloud = make_cudaq_executor(
        [ResultSpec({"011": 10}, (0, 1, 2), status_ready=True)],
        size=3,
    )

    result = executor.sample(state_110, shots_count=10)

    assert dict(result) == {"110": 10}
    assert result.probability("110") == 1.0
    assert executor.last_task_id == "mock-cudaq-task-1"
    assert executor.last_job is not None
    assert executor.last_job.status().state is JobState.DONE
    assert executor.last_qcis is not None
    assert "M Q0" in executor.last_qcis
    assert cloud.calls[0][2] == 10


def test_async_job_exposes_task_status_wait_and_cached_result() -> None:
    executor, cloud = make_cudaq_executor([ResultSpec({"0": 4}, (0,))], size=1)

    job = executor.sample_async(zero, shots_count=4)

    assert job.task_id == "mock-cudaq-task-1"
    assert job.task_ids == ("mock-cudaq-task-1",)
    assert job.status().state is JobState.SUBMITTED
    assert cloud.handles[0].wait_calls == []
    first = job.get(timeout=2, poll_interval=0.01)
    second = job.result()
    assert dict(first) == {"0": 4}
    assert first is second
    assert len(cloud.handles[0].wait_calls) == 1


def test_device_information_and_availability_are_exposed() -> None:
    executor, cloud = make_cudaq_executor([], size=3)

    assert executor.target.num_qubits == 3
    assert executor.target.couplings
    cloud.status = "calibration"
    cloud._available = False
    assert not executor.is_available()
    assert executor.target.status == "calibration"


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
def test_executor_rejects_invalid_wait_configuration(
    keyword: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        make_cudaq_executor([], size=1, **{keyword: value})


def test_unavailable_device_and_invalid_shots_fail_before_submission() -> None:
    executor, cloud = make_cudaq_executor(
        [ResultSpec({"0": 1}, (0,))],
        size=1,
        available=False,
    )

    with pytest.raises(AdapterDeviceError, match="not available"):
        executor.sample(zero, shots_count=1)
    assert cloud.calls == []
    with pytest.raises(ValueError, match="positive integer"):
        executor.sample(zero, shots_count=0)
