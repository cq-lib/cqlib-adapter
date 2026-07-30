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

"""PennyLane 0.45 Device backed by cqlib compilation and Tianyan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScript
from pennylane.transforms.core import CompilePipeline

from cqlib_adapter.common import (
    AdapterConversionError,
    CalibrationMode,
    CircuitCompiler,
    CompilationMode,
    CompilationOptions,
    DeviceStatus,
    NormalizedDevice,
    RunOptions,
    TianyanConnector,
    require_positive_finite,
)

from .converter import SUPPORTED_OPERATION_NAMES, pennylane_to_cqlib, supports_measurement
from .execution import PennyLaneExecution


def _supports_operation(operation: Any) -> bool:
    return str(operation.name) in SUPPORTED_OPERATION_NAMES


def _sample_measurement(measurement: Any) -> bool:
    return supports_measurement(measurement)


def _analytic_measurement(measurement: Any) -> bool:
    del measurement
    return False


_FACTORY_DEVICE_OPTIONS = frozenset(
    {
        "compiler",
        "timeout",
        "poll_interval",
        "calibration",
        "require_available",
        "compilation_mode",
        "initial_layout",
        "resource_policy",
        "seed",
    }
)


def _validate_factory_device_options(device_options: Mapping[str, Any]) -> None:
    """Reject misspelled Device options before authenticating with Tianyan."""

    unsupported = sorted(set(device_options).difference(_FACTORY_DEVICE_OPTIONS))
    if unsupported:
        raise TypeError(f"unsupported TianyanDevice factory options: {unsupported}")


def _split_authentication_options(
    explicit: Mapping[str, Any] | None,
    device_options: dict[str, Any],
    *,
    names: frozenset[str],
    option_name: str,
) -> dict[str, Any]:
    """Keep connector options separate while accepting legacy keyword forms."""

    if explicit is not None and not isinstance(explicit, Mapping):
        raise TypeError(f"{option_name} must be a mapping")
    authentication = dict(explicit or {})
    legacy = {name: device_options.pop(name) for name in names if name in device_options}
    duplicate = sorted(set(authentication).intersection(legacy))
    if duplicate:
        raise TypeError(f"{option_name} duplicates legacy authentication options: {duplicate}")
    authentication.update(legacy)
    unsupported = sorted(repr(name) for name in set(authentication).difference(names))
    if unsupported:
        raise TypeError(f"{option_name} contains unsupported authentication options: {unsupported}")
    return authentication


class TianyanDevice(Device):
    """Finite-shot PennyLane Device executing compiled QCIS on Tianyan."""

    _device_name = "cqlib.tianyan"

    def __init__(
        self,
        connector: TianyanConnector,
        device: NormalizedDevice,
        *,
        wires: int | Sequence[Any] | None = None,
        shots: int | None = None,
        compiler: CircuitCompiler | None = None,
        timeout: float = 120.0,
        poll_interval: float = 5.0,
        calibration: str | CalibrationMode = CalibrationMode.AUTO,
        require_available: bool = True,
        compilation_mode: str | CompilationMode = CompilationMode.NORMAL,
        initial_layout: Any | None = None,
        resource_policy: Any | None = None,
        seed: int | None = None,
    ) -> None:
        logical_wires: int | Sequence[Any] = device.num_qubits if wires is None else wires
        wire_count = logical_wires if isinstance(logical_wires, int) else len(tuple(logical_wires))
        if isinstance(wire_count, bool) or wire_count <= 0:
            raise ValueError("wires must define at least one PennyLane wire")
        if wire_count > device.num_qubits:
            raise ValueError(
                f"PennyLane device requests {wire_count} wires but {device.name!r} "
                f"has only {device.num_qubits} qubits"
            )
        super().__init__(wires=logical_wires, shots=shots)
        if self.wires is None:
            raise ValueError("PennyLane wires could not be resolved")
        self._connector = connector
        self._device = device
        self._compiler = compiler or CircuitCompiler()
        self._timeout = require_positive_finite(timeout, name="timeout")
        self._poll_interval = require_positive_finite(poll_interval, name="poll_interval")
        self._calibration = CalibrationMode(str(calibration).strip().lower())
        self._require_available = bool(require_available)
        self._compilation_options = CompilationOptions(
            mode=CompilationMode(str(compilation_mode).strip().lower()),
            initial_layout=initial_layout,
            resource_policy=resource_policy,
            seed=seed,
        )
        self._last_executions: tuple[PennyLaneExecution, ...] = ()

    @property
    def name(self) -> str:
        return self._device_name

    @property
    def device(self) -> NormalizedDevice:
        """Normalized cqlib-tianyan device snapshot."""

        return self._device

    @property
    def device_status(self) -> DeviceStatus:
        return self._device.status

    @property
    def operations(self) -> frozenset[str]:
        return SUPPORTED_OPERATION_NAMES

    @property
    def last_executions(self) -> tuple[PennyLaneExecution, ...]:
        """Most recent submitted executions, preserving batch order."""

        return self._last_executions

    @property
    def last_task_ids(self) -> tuple[str, ...]:
        return tuple(task_id for item in self._last_executions for task_id in item.task_ids)

    @property
    def last_qcis(self) -> tuple[str, ...]:
        return tuple(item.qcis for item in self._last_executions)

    def refresh_device(self) -> NormalizedDevice:
        """Refresh mutable Tianyan status fields."""

        self._device = self._connector.refresh_device_state(self._device)
        return self._device

    def is_available(self, *, refresh: bool = True) -> bool:
        if refresh:
            self.refresh_device()
        return self._device.available

    @classmethod
    def from_connector(
        cls,
        connector: TianyanConnector,
        device_name: str,
        **kwargs: Any,
    ) -> TianyanDevice:
        """Create a Device from an authenticated or mocked connector."""

        return cls(connector, connector.resolve_device(device_name), **kwargs)

    @classmethod
    def login(
        cls,
        api_key: str,
        device_name: str,
        *,
        wires: int | Sequence[Any] | None = None,
        shots: int | None = None,
        save_credentials: bool = False,
        login_options: Mapping[str, Any] | None = None,
        **device_options: Any,
    ) -> TianyanDevice:
        """Authenticate with cqlib-tianyan and create a PennyLane Device."""

        authentication = _split_authentication_options(
            login_options,
            device_options,
            names=frozenset({"domain", "auto_refresh", "credentials_path", "save_credentials"}),
            option_name="login_options",
        )
        _validate_factory_device_options(device_options)
        if "save_credentials" in authentication:
            raise TypeError(
                "save_credentials must be passed as an explicit TianyanDevice.login argument"
            )
        connector = TianyanConnector.login(
            api_key,
            save_credentials=save_credentials,
            **authentication,
        )
        return cls.from_connector(
            connector,
            device_name,
            wires=wires,
            shots=shots,
            **device_options,
        )

    @classmethod
    def from_credentials(
        cls,
        device_name: str,
        *,
        wires: int | Sequence[Any] | None = None,
        shots: int | None = None,
        credential_options: Mapping[str, Any] | None = None,
        **device_options: Any,
    ) -> TianyanDevice:
        """Load saved cqlib-tianyan credentials and create a Device."""

        authentication = _split_authentication_options(
            credential_options,
            device_options,
            names=frozenset({"domain", "auto_refresh", "credentials_path", "save_credentials"}),
            option_name="credential_options",
        )
        _validate_factory_device_options(device_options)
        connector = TianyanConnector.from_credentials(**authentication)
        return cls.from_connector(
            connector,
            device_name,
            wires=wires,
            shots=shots,
            **device_options,
        )

    def preprocess_transforms(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> CompilePipeline:
        """Decompose a QNode to operations handled by the cqlib translator."""

        del execution_config
        from pennylane import transforms
        from pennylane.devices import preprocess

        program = CompilePipeline()
        program.add_transform(
            preprocess.decompose,
            stopping_condition=_supports_operation,
            stopping_condition_shots=_supports_operation,
            device_wires=self.wires,
            target_gates=set(SUPPORTED_OPERATION_NAMES),
            name=self.name,
            error=AdapterConversionError,
        )
        program.add_transform(transforms.broadcast_expand)
        program.add_transform(preprocess.validate_device_wires, self.wires, name=self.name)
        program.add_transform(
            preprocess.validate_measurements,
            analytic_measurements=_analytic_measurement,
            sample_measurements=_sample_measurement,
            name=self.name,
        )
        return program

    def _shots_for(self, tape: QuantumScript) -> int:
        if tape.shots.has_partitioned_shots:
            raise AdapterConversionError("PennyLane shot vectors are not supported")
        shots = tape.shots.total_shots
        if shots is None:
            shots = self.shots.total_shots
        if shots is None:
            raise AdapterConversionError(
                "TianyanDevice requires finite shots; use @qml.set_shots(shots)"
            )
        return int(shots)

    def _submit_one(self, tape: QuantumScript, circuit_index: int) -> PennyLaneExecution:
        shots = self._shots_for(tape)
        bundle = pennylane_to_cqlib(tape, wire_order=self.wires)
        artifact = self._compiler.compile(
            bundle,
            device=self._device,
            options=self._compilation_options,
            circuit_index=circuit_index,
        )
        run_options = RunOptions(
            device_name=self._device.name,
            shots=shots,
            timeout=self._timeout,
            poll_interval=self._poll_interval,
            calibration=self._calibration,
            require_available=self._require_available,
        )
        adapter_job = self._connector.submit((artifact,), options=run_options)
        return PennyLaneExecution(
            adapter_job,
            artifact,
            tape,
            tuple(self.wires),
        )

    @staticmethod
    def _circuit_batch(
        circuits: QuantumScript | Sequence[QuantumScript],
    ) -> tuple[bool, tuple[QuantumScript, ...]]:
        single = isinstance(circuits, QuantumScript)
        batch = (circuits,) if single else tuple(circuits)
        if not batch:
            raise ValueError("PennyLane execution requires at least one QuantumScript")
        if not all(isinstance(tape, QuantumScript) for tape in batch):
            raise TypeError("every circuit must be a pennylane.tape.QuantumScript")
        return single, batch

    def submit(
        self,
        circuits: QuantumScript | Sequence[QuantumScript],
    ) -> PennyLaneExecution | tuple[PennyLaneExecution, ...]:
        """Compile and submit without waiting, returning queryable tasks."""

        single, batch = self._circuit_batch(circuits)
        executions: list[PennyLaneExecution] = []
        self._last_executions = ()
        for index, tape in enumerate(batch):
            executions.append(self._submit_one(tape, index))
            self._last_executions = tuple(executions)
        return executions[0] if single else tuple(executions)

    def execute(
        self,
        circuits: QuantumScript | Sequence[QuantumScript],
        execution_config: ExecutionConfig | None = None,
    ) -> Any:
        """Compile, submit, wait and return PennyLane-native measurement data."""

        del execution_config
        single = isinstance(circuits, QuantumScript)
        batch = (circuits,) if single else tuple(circuits)
        if not batch:
            raise ValueError("PennyLane execute requires at least one QuantumScript")
        if not all(isinstance(tape, QuantumScript) for tape in batch):
            raise TypeError("every circuit must be a pennylane.tape.QuantumScript")
        values: list[Any] = []
        executions: list[PennyLaneExecution] = []
        for index, tape in enumerate(batch):
            execution = self._submit_one(tape, index)
            self._last_executions = (*executions, execution)
            value = execution.result(
                timeout=self._timeout,
                poll_interval=self._poll_interval,
            )
            values.append(value)
            executions.append(execution)
        self._last_executions = tuple(executions)
        if self.tracker.active:
            self.tracker.update(executions=len(batch), shots=sum(item.shots for item in executions))
            self.tracker.record()
        return values[0] if single else tuple(values)

    def __repr__(self) -> str:
        return (
            f"<TianyanDevice device={self._device.name!r} "
            f"wires={list(self.wires)!r} shots={self.shots.total_shots!r}>"
        )


__all__ = ["TianyanDevice"]
