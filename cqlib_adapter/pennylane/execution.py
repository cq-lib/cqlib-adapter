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

"""PennyLane execution wrapper around one common Tianyan job."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pennylane.tape import QuantumScript

from cqlib_adapter.common import AdapterJob, CompilationArtifact, JobSnapshot

from .result import canonical_to_pennylane_result


@dataclass(slots=True)
class PennyLaneExecution:
    """One submitted QuantumScript with task, QCIS and PennyLane result access."""

    adapter_job: AdapterJob
    artifact: CompilationArtifact
    tape: QuantumScript
    wire_order: tuple[Any, ...]
    _result: Any = field(default=None, init=False, repr=False)
    _resolved: bool = field(default=False, init=False, repr=False)

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
    def shots(self) -> int:
        return self.adapter_job.shots

    def status(self) -> JobSnapshot:
        return self.adapter_job.status()

    def result(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> Any:
        """Wait and return PennyLane-native counts/probs/sample values."""

        if not self._resolved:
            canonical = self.adapter_job.result(
                timeout=timeout,
                poll_interval=poll_interval,
            )
            if len(canonical) != 1:
                raise RuntimeError("one PennyLane execution must resolve to exactly one result")
            self._result = canonical_to_pennylane_result(
                canonical[0],
                self.tape.measurements,
                wire_order=self.wire_order,
                active_wires=tuple(self.tape.wires),
            )
            self._resolved = True
        return self._result


__all__ = ["PennyLaneExecution"]
