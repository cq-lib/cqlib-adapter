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

"""Structural protocols testable without native or cloud packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


class QubitLike(Protocol):
    @property
    def index(self) -> int: ...


class InstructionLike(Protocol):
    @property
    def name(self) -> str: ...


class ValueInstructionLike(Protocol):
    @property
    def instruction(self) -> InstructionLike | None: ...


class OperationLike(Protocol):
    @property
    def instruction(self) -> ValueInstructionLike: ...

    @property
    def qubits(self) -> Sequence[QubitLike]: ...


class CircuitLike(Protocol):
    @property
    def global_phase(self) -> Any: ...
    @property
    def num_qubits(self) -> int: ...

    @property
    def qubits(self) -> Sequence[QubitLike]: ...

    @property
    def operations(self) -> Sequence[OperationLike]: ...

    def validate(self) -> None: ...


class TopologyLike(Protocol):
    @property
    def qubits(self) -> Sequence[QubitLike]: ...

    def successors(self, qubit: QubitLike) -> Sequence[QubitLike]: ...


class DeviceLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def qubits(self) -> Sequence[QubitLike]: ...

    @property
    def invalid_qubits(self) -> Sequence[QubitLike]: ...

    @property
    def native_gates(self) -> Sequence[Any]: ...

    @property
    def topology(self) -> TopologyLike: ...


class ExecutionStatusLike(Protocol):
    @property
    def kind(self) -> Any: ...

    @property
    def error_msg(self) -> str | None: ...

    @property
    def error_code(self) -> str | int | None: ...


class ExecutionResultLike(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def shots(self) -> int: ...

    @property
    def num_qubits(self) -> int: ...

    @property
    def qubits(self) -> Sequence[QubitLike]: ...

    @property
    def status(self) -> ExecutionStatusLike: ...

    @property
    def counts(self) -> Mapping[str, int]: ...

    @property
    def probabilities(self) -> Mapping[str, float] | None: ...


@runtime_checkable
class TaskHandleLike(Protocol):
    @property
    def task_ids(self) -> Sequence[str]: ...

    @property
    def device_name(self) -> str: ...

    @property
    def shots(self) -> int: ...

    def status(self) -> Sequence[ExecutionResultLike]: ...

    def wait(
        self,
        timeout_secs: float | None = None,
        poll_interval_secs: float = 5.0,
    ) -> Sequence[ExecutionResultLike]: ...


class BackendLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def status(self) -> Any: ...

    @property
    def toll(self) -> Any: ...

    def is_available(self) -> bool: ...

    def num_qubits(self) -> int: ...

    def device_config(self) -> DeviceLike: ...

    def run(self, circuits: Sequence[str], shots: int = 1024) -> TaskHandleLike: ...

    def run_raw(self, circuits: Sequence[str], shots: int = 1024) -> TaskHandleLike: ...

    def run_with_mode(
        self,
        circuits: Sequence[str],
        shots: int = 1024,
        mode: str = "auto",
    ) -> TaskHandleLike: ...


class PlatformLike(Protocol):
    def list_backends(self) -> Sequence[BackendLike]: ...

    def get_backend(self, name: str) -> BackendLike: ...


class CompileResultLike(Protocol):
    @property
    def circuit(self) -> CircuitLike: ...

    @property
    def changed(self) -> bool: ...

    @property
    def steps(self) -> Sequence[Any]: ...


class CqlibRuntime(Protocol):
    """Minimal lazy-loaded cqlib surface used by the common compiler."""

    def normal_mode(self) -> Any: ...

    def enhanced_mode(self) -> Any: ...

    def compile(
        self,
        circuit: CircuitLike,
        *,
        mode: Any,
        target_basis: Sequence[str] | None,
        device: DeviceLike | None,
        initial_layout: Any | None,
        resource_policy: Any | None,
        seed: int | None,
    ) -> CompileResultLike: ...

    def dumps(self, circuit: CircuitLike) -> str: ...

    def loads(self, qcis: str) -> CircuitLike: ...


__all__ = [
    "BackendLike",
    "CircuitLike",
    "CompileResultLike",
    "CqlibRuntime",
    "DeviceLike",
    "ExecutionResultLike",
    "ExecutionStatusLike",
    "InstructionLike",
    "OperationLike",
    "PlatformLike",
    "QubitLike",
    "TaskHandleLike",
    "TopologyLike",
    "ValueInstructionLike",
]
