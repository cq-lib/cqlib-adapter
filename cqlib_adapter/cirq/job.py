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

"""Cirq execution wrapper around one common Tianyan job."""

from __future__ import annotations

from dataclasses import dataclass, field

import cirq

from cqlib_adapter.common import AdapterJob, CanonicalResult, CompilationArtifact, JobSnapshot

from .result import canonical_to_cirq_probabilities, canonical_to_cirq_result


@dataclass(slots=True)
class CirqExecution:
    """One submitted Cirq parameter resolution with task and result access."""

    adapter_job: AdapterJob
    artifact: CompilationArtifact
    resolver: cirq.ParamResolver
    _canonical: CanonicalResult | None = field(default=None, init=False, repr=False)
    _result: cirq.ResultDict | None = field(default=None, init=False, repr=False)

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
    def repetitions(self) -> int:
        return self.adapter_job.shots

    def status(self) -> JobSnapshot:
        return self.adapter_job.status()

    def _canonical_result(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> CanonicalResult:
        if self._canonical is None:
            canonical = self.adapter_job.result(
                timeout=timeout,
                poll_interval=poll_interval,
            )
            if len(canonical) != 1:
                raise RuntimeError("one Cirq execution must resolve to exactly one result")
            self._canonical = canonical[0]
        return self._canonical

    def probabilities(
        self,
        key: str | None = None,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> dict[int, float]:
        """Wait if necessary and return probabilities for one measurement key."""

        return canonical_to_cirq_probabilities(
            self._canonical_result(
                timeout=timeout,
                poll_interval=poll_interval,
            ),
            self.artifact.metadata,
            key=key,
        )

    def result(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> cirq.ResultDict:
        """Wait and return the standard Cirq ResultDict."""

        if self._result is None:
            self._result = canonical_to_cirq_result(
                self._canonical_result(
                    timeout=timeout,
                    poll_interval=poll_interval,
                ),
                self.artifact.metadata,
                params=self.resolver,
            )
        return self._result


__all__ = ["CirqExecution"]
