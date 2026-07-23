from __future__ import annotations

import numpy as np
import pytest

cudaq = pytest.importorskip("cudaq")
pytest.importorskip("cqlib")

from cqlib_adapter.common import AdapterDeviceError, JobState  # noqa: E402
from cqlib_adapter.cudaq import CqlibSimulator  # noqa: E402
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


@cudaq.kernel
def x_basis_plus() -> None:
    qubit = cudaq.qubit()
    h(qubit)  # noqa: F821  # Prepare |+>.
    h(qubit)  # noqa: F821  # Rotate X basis to Z before mz.
    mz(qubit)  # noqa: F821


@cudaq.kernel
def grover_11() -> None:
    qubits = cudaq.qvector(2)
    h(qubits)  # noqa: F821
    z.ctrl(qubits[0], qubits[1])  # noqa: F821
    h(qubits)  # noqa: F821
    x(qubits)  # noqa: F821
    z.ctrl(qubits[0], qubits[1])  # noqa: F821
    x(qubits)  # noqa: F821
    h(qubits)  # noqa: F821
    mz(qubits)  # noqa: F821


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


def test_local_cqlib_simulator_preserves_cudaq_110_order() -> None:
    simulator = CqlibSimulator(3)

    result = simulator.sample(state_110, shots_count=32)

    assert dict(result) == {"110": 32}
    assert simulator.simulator_calls
    assert simulator.last_task_id == "local-cudaq-task-1"


def test_local_cqlib_simulator_grover_finds_marked_state() -> None:
    simulator = CqlibSimulator(2)

    result = simulator.sample(grover_11, shots_count=64)

    assert dict(result) == {"11": 64}
    assert simulator.last_qcis is not None
    assert "CZ Q0 Q1" in simulator.last_qcis


def test_local_cqlib_simulator_supports_measurement_only_zero_state() -> None:
    assert dict(CqlibSimulator(1).sample(zero, shots_count=12)) == {"0": 12}


def test_local_cqlib_simulator_supports_explicit_x_basis_measurement() -> None:
    assert dict(CqlibSimulator(1).sample(x_basis_plus, shots_count=12)) == {"0": 12}


def test_local_cqlib_statevector_matches_cudaq_up_to_global_phase() -> None:
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(3)
    kernel.ry(0.37, qubits[0])
    kernel.rz(-0.29, qubits[1])
    kernel.cx(qubits[0], qubits[2])
    kernel.ry(0.61, qubits[2])

    reference = np.asarray(cudaq.get_state(kernel).to_numpy(), dtype=np.complex128)
    simulator = CqlibSimulator(3)
    result = simulator.run_statevector(kernel)
    candidate = np.asarray(result.data, dtype=np.complex128)
    overlap = np.vdot(reference, candidate)
    fidelity = abs(overlap) ** 2 / (
        float(np.vdot(reference, reference).real) * float(np.vdot(candidate, candidate).real)
    )

    assert fidelity == pytest.approx(1.0, abs=1e-12)
    assert "CZ" in result.qcis
    assert "CX" not in result.qcis
    assert all(line.split()[0].upper() != "M" for line in result.qcis.splitlines())
    assert set(result.physical_qubits) == {0, 1, 2}
    assert simulator.simulator_calls == ()


def test_local_cqlib_statevector_uses_cudaq_little_endian_indices() -> None:
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(3)
    kernel.x(qubits[0])
    kernel.x(qubits[1])

    result = CqlibSimulator(3).run_statevector(kernel)

    assert np.argmax(np.abs(result.data)) == 3
    assert abs(result.data[3]) == pytest.approx(1.0, abs=1e-12)


def test_local_cqlib_statevector_rejects_width_mismatch() -> None:
    kernel = cudaq.make_kernel()
    kernel.qalloc(2)

    with pytest.raises(ValueError, match="width must match"):
        CqlibSimulator(3).run_statevector(kernel)


def test_local_cqlib_simulator_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CqlibSimulator(0)
