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

from cqlib_adapter.common import (
    AdapterDeviceError,
    AdapterJob,
    AdapterJobError,
    AdapterJobTimeoutError,
    AdapterResultError,
    AdapterSubmissionError,
    CalibrationMode,
    CircuitCompiler,
    CompilationArtifact,
    CompiledMeasurement,
    JobState,
    MeasurementMetadata,
    MeasurementSlot,
    NormalizedDevice,
    RunOptions,
    TianyanConnector,
    TranslationBundle,
    TranslationMetadata,
)

from .fakes import (
    FakeBackend,
    FakeCircuit,
    FakeExecutionResult,
    FakeHandle,
    FakePlatform,
    FakeRuntime,
    FakeStatus,
)

pytestmark = pytest.mark.unit


def _artifact(device: NormalizedDevice, label: str = "q0") -> CompilationArtifact:
    metadata = TranslationMetadata(
        "qiskit",
        (label,),
        MeasurementMetadata((MeasurementSlot(label, 0),), 1),
    )
    return CompilationArtifact(
        "M Q0",
        FakeCircuit((("measure_bit", (0,)),)),
        metadata,
        device,
        False,
        (),
        (CompiledMeasurement(0, 0),),
    )


def _backend_with_result(
    *,
    task_id: str = "task-1",
    status_results: list[FakeExecutionResult] | None = None,
) -> tuple[FakeBackend, FakeHandle]:
    result = FakeExecutionResult(task_id, 10, (0,), {"0": 6, "1": 4})
    handle = FakeHandle(
        [task_id],
        [result],
        status_results=list(status_results or []),
        shots=10,
    )
    return FakeBackend(handles=[handle]), handle


def test_connector_refreshes_only_mutable_device_state() -> None:
    backend = FakeBackend()
    connector = TianyanConnector(FakePlatform([backend]))
    original = connector.get_device("fake")
    backend.display_name = "renamed display"
    backend.status = "calibration"
    backend.toll = "paid"
    backend._available = False

    refreshed = connector.refresh_device_state(original)

    assert original.status.value == "running"
    assert refreshed.display_name == "renamed display"
    assert refreshed.status.value == "calibration"
    assert refreshed.toll.value == "paid"
    assert not refreshed.available
    assert refreshed.cqlib_device is original.cqlib_device


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (CalibrationMode.AUTO, "auto"),
        (CalibrationMode.ENABLED, "enabled"),
        (CalibrationMode.DISABLED, "disabled"),
    ],
)
def test_connector_selects_calibration_submission_path(
    mode: CalibrationMode,
    expected: str,
) -> None:
    backend, _ = _backend_with_result()
    connector = TianyanConnector(FakePlatform([backend]))
    device = connector.get_device("fake")
    job = connector.submit(
        [_artifact(device)],
        options=RunOptions("fake", shots=10, calibration=mode),
    )
    assert job.task_ids == ("task-1",)
    assert backend.calls == [(expected, ["M Q0"], 10)]


def test_connector_propagates_wait_defaults_to_job() -> None:
    backend, handle = _backend_with_result()
    connector = TianyanConnector(FakePlatform([backend]))
    device = connector.get_device("fake")
    job = connector.submit(
        [_artifact(device)],
        options=RunOptions("fake", shots=10, timeout=9.0, poll_interval=0.25),
    )
    job.wait()
    passed_timeout, passed_poll = handle.wait_calls[0]
    assert passed_timeout is not None
    assert 0 < passed_timeout <= 9.0
    assert passed_poll == 0.25


def test_connector_rejects_offline_device() -> None:
    backend = FakeBackend(available=False)
    connector = TianyanConnector(FakePlatform([backend]))
    device = connector.get_device("fake")
    with pytest.raises(AdapterDeviceError, match="not available"):
        connector.submit([_artifact(device)], options=RunOptions("fake"))


def test_connector_allows_explicit_offline_override() -> None:
    backend = FakeBackend(available=False)
    connector = TianyanConnector(FakePlatform([backend]))
    device = connector.get_device("fake")
    job = connector.submit(
        [_artifact(device)],
        options=RunOptions("fake", require_available=False),
    )
    assert job.task_ids == ("task-1",)


def test_connector_rejects_artifact_for_different_device() -> None:
    fake = FakeBackend(name="fake")
    other = FakeBackend(name="other")
    connector = TianyanConnector(FakePlatform([fake, other]))
    artifact = _artifact(connector.get_device("other"))
    with pytest.raises(AdapterSubmissionError, match="not compiled"):
        connector.submit([artifact], options=RunOptions("fake"))


def test_list_devices_and_missing_device_are_wrapped() -> None:
    connector = TianyanConnector(FakePlatform([FakeBackend()]))
    assert [device.name for device in connector.list_devices()] == ["fake"]
    with pytest.raises(AdapterDeviceError, match="failed to get"):
        connector.get_device("missing")


@pytest.mark.parametrize(
    ("ready", "expected"),
    [
        ([], JobState.SUBMITTED),
        ([FakeExecutionResult("a", 10, (0,), {"0": 10})], JobState.PARTIAL),
        (
            [
                FakeExecutionResult("a", 10, (0,), {"0": 10}),
                FakeExecutionResult("b", 10, (0,), {"1": 10}),
            ],
            JobState.DONE,
        ),
    ],
)
def test_job_status_uses_ready_result_count(
    ready: list[FakeExecutionResult],
    expected: JobState,
) -> None:
    device = NormalizedDevice.from_backend(FakeBackend())
    handle = FakeHandle(["a", "b"], [], status_results=ready, shots=10)
    snapshot = AdapterJob([handle], [_artifact(device), _artifact(device)]).status()
    assert snapshot.state is expected
    assert snapshot.ready_count == len(ready)
    assert snapshot.total_count == 2


def test_job_status_reports_failed_terminal_result() -> None:
    failed = FakeExecutionResult(
        "a",
        10,
        (0,),
        {},
        status=FakeStatus("failed", "boom", 1),
    )
    device = NormalizedDevice.from_backend(FakeBackend())
    handle = FakeHandle(["a"], [], status_results=[failed], shots=10)
    assert AdapterJob([handle], [_artifact(device)]).status().state is JobState.ERROR


def test_job_wait_restores_task_order_and_caches_results() -> None:
    device = NormalizedDevice.from_backend(FakeBackend())
    result_a = FakeExecutionResult("a", 10, (0,), {"0": 10})
    result_b = FakeExecutionResult("b", 10, (0,), {"1": 10})
    handle = FakeHandle(["a", "b"], [result_b, result_a], shots=10)
    job = AdapterJob([handle], [_artifact(device), _artifact(device)])
    first = job.wait(timeout=5, poll_interval=0.1)
    second = job.result(timeout=5)
    assert [result.task_id for result in first] == ["a", "b"]
    assert first[0].counts == {"0": 10}
    assert first[1].counts == {"1": 10}
    assert second is first
    assert len(handle.wait_calls) == 1
    assert job.status().state is JobState.DONE


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([FakeExecutionResult("a", 10, (0,), {"0": 10})], "missing"),
        (
            [
                FakeExecutionResult("a", 10, (0,), {"0": 10}),
                FakeExecutionResult("a", 10, (0,), {"0": 10}),
            ],
            "duplicate",
        ),
        (
            [
                FakeExecutionResult("a", 10, (0,), {"0": 10}),
                FakeExecutionResult("unknown", 10, (0,), {"0": 10}),
            ],
            "unknown",
        ),
    ],
)
def test_job_rejects_incomplete_or_ambiguous_batch_results(
    results: list[FakeExecutionResult],
    message: str,
) -> None:
    device = NormalizedDevice.from_backend(FakeBackend())
    handle = FakeHandle(["a", "b"], results, shots=10)
    with pytest.raises(AdapterResultError, match=message):
        AdapterJob([handle], [_artifact(device), _artifact(device)]).wait()


def test_job_wraps_native_timeout() -> None:
    device = NormalizedDevice.from_backend(FakeBackend())
    handle = FakeHandle(
        ["a"],
        [],
        wait_error=TimeoutError("native timeout"),
        shots=10,
    )
    with pytest.raises(AdapterJobTimeoutError, match="native timeout"):
        AdapterJob([handle], [_artifact(device)]).wait(timeout=1)


def test_job_rejects_inconsistent_handle_shots() -> None:
    device = NormalizedDevice.from_backend(FakeBackend())
    first = FakeHandle(["a"], [], shots=10)
    second = FakeHandle(["b"], [], shots=20)
    job = AdapterJob([first, second], [_artifact(device), _artifact(device)])
    with pytest.raises(AdapterJobError, match="inconsistent"):
        _ = job.shots


def test_compile_submit_wait_convert_closed_loop() -> None:
    result = FakeExecutionResult("closed-loop", 10, (0,), {"0": 7, "1": 3})
    handle = FakeHandle(["closed-loop"], [result], shots=10)
    backend = FakeBackend(handles=[handle])
    compiled = FakeCircuit((("X90", (0,)), ("measure_bit", (0,))))
    connector = TianyanConnector(
        FakePlatform([backend]),
        compiler=CircuitCompiler(FakeRuntime(compiled)),
    )
    metadata = TranslationMetadata(
        "qiskit",
        ("q0",),
        MeasurementMetadata((MeasurementSlot("q0", 0),), 1),
    )
    job = connector.compile_and_submit(
        [TranslationBundle(FakeCircuit((("H", (0,)),)), metadata)],
        run_options=RunOptions("fake", shots=10),
    )
    converted = job.wait(timeout=5)
    assert converted[0].task_id == "closed-loop"
    assert converted[0].counts == {"0": 7, "1": 3}
    assert backend.calls[0][1][0] == "X90 Q0\nM Q0"


def test_partial_submission_error_reports_already_created_task_ids() -> None:
    class FailingBackend(FakeBackend):
        def run(self, circuits: list[str], shots: int = 1024) -> FakeHandle:
            if self.calls:
                raise RuntimeError("second request failed")
            return super().run(circuits, shots)

    first = FakeHandle(["created-task"], [], shots=10)
    backend = FailingBackend(handles=[first])
    connector = TianyanConnector(FakePlatform([backend]))
    device = connector.get_device("fake")
    with pytest.raises(AdapterSubmissionError) as raised:
        connector.submit(
            [_artifact(device, "q0"), _artifact(device, "q1")],
            options=RunOptions("fake", shots=10),
        )
    assert "already submitted task IDs: ['created-task']" in str(raised.value)
    assert "circuit_index=1" in str(raised.value)


def test_resolve_device_skips_unrelated_backend_with_broken_config() -> None:
    class BrokenBackend(FakeBackend):
        def device_config(self):
            raise RuntimeError("download failed")

    broken = BrokenBackend(name="supremacy_sample")
    target = FakeBackend(name="tianyan-176-2")
    target.display_name = "天衍176-2"
    connector = TianyanConnector(FakePlatform([broken, target]))

    by_display = connector.resolve_device("天衍176-2")
    by_name = connector.resolve_device("tianyan-176-2")

    assert by_display.name == "tianyan-176-2"
    assert by_name.display_name == "天衍176-2"


def test_resolve_device_reports_missing_and_ambiguous_selectors() -> None:
    first = FakeBackend(name="first")
    second = FakeBackend(name="second")
    first.display_name = second.display_name = "same display"
    connector = TianyanConnector(FakePlatform([first, second]))

    with pytest.raises(AdapterDeviceError, match="was not found"):
        connector.resolve_device("missing")
    with pytest.raises(AdapterDeviceError, match="ambiguous"):
        connector.resolve_device("same display")
