"""CUDA-Q asynchronous execution wrapper around one common adapter job."""

from __future__ import annotations

from dataclasses import dataclass, field

from cqlib_adapter.common import AdapterJob, CompilationArtifact, JobSnapshot

from .result import CudaQSampleResult, canonical_to_cudaq_result


@dataclass(slots=True)
class CudaQJob:
    """One already-submitted CUDA-Q kernel with query and wait methods."""

    adapter_job: AdapterJob
    artifact: CompilationArtifact
    _result: CudaQSampleResult | None = field(default=None, init=False, repr=False)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.adapter_job.task_ids

    @property
    def task_id(self) -> str:
        return self.task_ids[0]

    @property
    def qcis(self) -> str:
        return self.artifact.qcis

    @property
    def shots_count(self) -> int:
        return self.adapter_job.shots

    @property
    def device_name(self) -> str:
        return self.adapter_job.device_name

    def status(self) -> JobSnapshot:
        return self.adapter_job.status()

    def result(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> CudaQSampleResult:
        """Wait if necessary and return CUDA-Q-compatible sample counts."""

        if self._result is None:
            canonical = self.adapter_job.result(
                timeout=timeout,
                poll_interval=poll_interval,
            )
            if len(canonical) != 1:
                raise RuntimeError("one CUDA-Q job must resolve to exactly one result")
            self._result = canonical_to_cudaq_result(canonical[0])
        return self._result

    def get(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> CudaQSampleResult:
        """CUDA-Q ``AsyncSampleResult.get``-style alias for :meth:`result`."""

        return self.result(timeout=timeout, poll_interval=poll_interval)

    def wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> CudaQSampleResult:
        return self.result(timeout=timeout, poll_interval=poll_interval)


__all__ = ["CudaQJob"]
