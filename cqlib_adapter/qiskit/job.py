"""Qiskit JobV1 wrapper around one framework-neutral Tianyan job."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from qiskit.providers import JobStatus, JobV1
from qiskit.providers.exceptions import JobError
from qiskit.result import Result

from cqlib_adapter.common import AdapterJob, CompilationArtifact, JobState

from .result import canonical_to_qiskit_result

if TYPE_CHECKING:
    from .backend import TianyanBackend


class TianyanJob(JobV1):
    """Submitted Tianyan work with Qiskit-compatible status and result methods."""

    def __init__(
        self,
        backend: TianyanBackend,
        adapter_job: AdapterJob,
        artifacts: Sequence[CompilationArtifact],
    ) -> None:
        self._adapter_job = adapter_job
        self._artifacts = tuple(artifacts)
        self._qiskit_result: Result | None = None
        super().__init__(backend, ",".join(adapter_job.task_ids))

    @property
    def task_ids(self) -> tuple[str, ...]:
        """All Tianyan task IDs, preserving submitted-circuit order."""

        return self._adapter_job.task_ids

    @property
    def compilation_artifacts(self) -> tuple[CompilationArtifact, ...]:
        """Compiled QCIS artifacts in submitted-circuit order."""

        return self._artifacts

    @property
    def qcis(self) -> tuple[str, ...]:
        """QCIS payloads actually submitted to cqlib-tianyan."""

        return tuple(artifact.qcis for artifact in self._artifacts)

    def submit(self) -> None:
        """Reject a second submission because Backend.run submits immediately."""

        raise JobError("TianyanJob is already submitted by backend.run()")

    def status(self) -> JobStatus:
        """Map the Tianyan aggregate state to Qiskit's JobStatus."""

        state = self._adapter_job.status().state
        return {
            JobState.SUBMITTED: JobStatus.QUEUED,
            JobState.PARTIAL: JobStatus.RUNNING,
            JobState.DONE: JobStatus.DONE,
            JobState.ERROR: JobStatus.ERROR,
        }[state]

    def result(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float | None = None,
    ) -> Result:
        """Wait for Tianyan and return a standard Qiskit Result."""

        if self._qiskit_result is None:
            canonical = self._adapter_job.result(
                timeout=timeout,
                poll_interval=poll_interval,
            )
            backend = self.backend()
            self._qiskit_result = canonical_to_qiskit_result(
                canonical,
                self._artifacts,
                backend_name=backend.name,
                backend_version=backend.backend_version,
            )
        return self._qiskit_result

    def cancel(self) -> bool:
        """Return False because cqlib-tianyan does not expose task cancellation."""

        return False


__all__ = ["TianyanJob"]
