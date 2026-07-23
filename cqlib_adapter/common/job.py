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

"""Framework-independent Tianyan task state and waiting model."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .compiler import CompilationArtifact
from .errors import (
    AdapterJobError,
    AdapterJobTimeoutError,
    AdapterResultError,
    ErrorContext,
)
from .result import CanonicalResult, ResultConverter
from .typing import ExecutionResultLike, TaskHandleLike


class JobState(StrEnum):
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    state: JobState
    task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]

    @property
    def ready_count(self) -> int:
        return len(self.completed_task_ids)

    @property
    def total_count(self) -> int:
        return len(self.task_ids)


class AdapterJob:
    """A stable batch facade over one or more Tianyan task handles."""

    def __init__(
        self,
        handles: Sequence[TaskHandleLike],
        artifacts: Sequence[CompilationArtifact],
        *,
        converter: ResultConverter | None = None,
        default_timeout: float = 120.0,
        default_poll_interval: float = 5.0,
    ) -> None:
        if not handles:
            raise ValueError("at least one task handle is required")
        task_ids = tuple(task_id for handle in handles for task_id in handle.task_ids)
        if len(task_ids) != len(artifacts):
            raise ValueError("task ID count must match compilation artifact count")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task IDs must be unique")
        if default_timeout <= 0 or default_poll_interval <= 0:
            raise ValueError("default wait values must be positive")
        self._handles = tuple(handles)
        self._artifacts = dict(zip(task_ids, artifacts, strict=True))
        self._task_ids = task_ids
        self._converter = converter or ResultConverter()
        self._default_timeout = default_timeout
        self._default_poll_interval = default_poll_interval
        self._cached: tuple[CanonicalResult, ...] | None = None

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self._task_ids

    @property
    def device_name(self) -> str:
        names = {handle.device_name for handle in self._handles}
        return next(iter(names)) if len(names) == 1 else ",".join(sorted(names))

    @property
    def shots(self) -> int:
        shots = {handle.shots for handle in self._handles}
        if len(shots) != 1:
            raise AdapterJobError("batch contains inconsistent shot counts")
        return next(iter(shots))

    def status(self) -> JobSnapshot:
        if self._cached is not None:
            return JobSnapshot(JobState.DONE, self.task_ids, self.task_ids)
        try:
            ready = [result for handle in self._handles for result in handle.status()]
            by_id = self._index_results(ready, require_all=False)
            kinds = {
                str(result.status.kind).strip().lower().rsplit(".", 1)[-1]
                for result in by_id.values()
            }
            if kinds.intersection({"failed", "cancelled", "error"}):
                state = JobState.ERROR
            elif len(by_id) == len(self._task_ids):
                state = JobState.DONE
            elif by_id:
                state = JobState.PARTIAL
            else:
                state = JobState.SUBMITTED
            completed = tuple(task_id for task_id in self._task_ids if task_id in by_id)
            return JobSnapshot(state, self.task_ids, completed)
        except AdapterJobError:
            raise
        except Exception as exc:
            raise AdapterJobError(f"failed to query Tianyan task status: {exc}") from exc

    def wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> tuple[CanonicalResult, ...]:
        if self._cached is not None:
            return self._cached
        effective_timeout = self._default_timeout if timeout is None else timeout
        effective_poll = self._default_poll_interval if poll_interval is None else poll_interval
        if effective_timeout <= 0:
            raise ValueError("timeout must be positive")
        if effective_poll <= 0:
            raise ValueError("poll_interval must be positive")
        started = time.monotonic()
        raw: list[ExecutionResultLike] = []
        try:
            for handle in self._handles:
                remaining = effective_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise AdapterJobTimeoutError(
                        "waiting for Tianyan tasks timed out",
                        context=ErrorContext(device_name=self.device_name),
                    )
                raw.extend(
                    handle.wait(
                        timeout_secs=remaining,
                        poll_interval_secs=effective_poll,
                    )
                )
        except AdapterJobTimeoutError:
            raise
        except Exception as exc:
            message = str(exc)
            context = ErrorContext(device_name=self.device_name)
            if isinstance(exc, TimeoutError) or "timeout" in message.lower():
                raise AdapterJobTimeoutError(message, context=context) from exc
            raise AdapterJobError(
                f"failed while waiting for Tianyan tasks: {message}",
                context=context,
            ) from exc

        by_id = self._index_results(raw, require_all=True)
        self._cached = tuple(
            self._converter.convert(by_id[task_id], self._artifacts[task_id])
            for task_id in self._task_ids
        )
        return self._cached

    def result(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> tuple[CanonicalResult, ...]:
        return self.wait(timeout=timeout, poll_interval=poll_interval)

    def _index_results(
        self,
        results: Sequence[ExecutionResultLike],
        *,
        require_all: bool,
    ) -> dict[str, ExecutionResultLike]:
        indexed: dict[str, ExecutionResultLike] = {}
        known = set(self._task_ids)
        for result in results:
            if result.task_id not in known:
                raise AdapterResultError(
                    "Tianyan returned a result for an unknown task",
                    context=ErrorContext(task_id=result.task_id),
                )
            if result.task_id in indexed:
                raise AdapterResultError(
                    "Tianyan returned duplicate results for one task",
                    context=ErrorContext(task_id=result.task_id),
                )
            indexed[result.task_id] = result
        if require_all and set(indexed) != known:
            missing = sorted(known - set(indexed))
            raise AdapterResultError(f"Tianyan response is missing task results: {missing}")
        return indexed


__all__ = ["AdapterJob", "JobSnapshot", "JobState"]
