"""Cirq Sampler backed by cqlib compilation and Tianyan execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cirq

from cqlib_adapter.common import (
    CalibrationMode,
    CircuitCompiler,
    CompilationMode,
    CompilationOptions,
    NormalizedDevice,
    RunOptions,
    TianyanConnector,
)

from .converter import cirq_to_cqlib
from .device import TianyanDevice
from .job import CirqExecution


class TianyanSampler(cirq.Sampler):
    """Standard Cirq Sampler executing resolved circuits through Tianyan."""

    def __init__(
        self,
        connector: TianyanConnector,
        device: NormalizedDevice,
        *,
        compiler: CircuitCompiler | None = None,
        timeout: float = 120.0,
        poll_interval: float = 5.0,
        calibration: str | CalibrationMode = CalibrationMode.AUTO,
        require_available: bool = True,
        compilation_mode: str | CompilationMode = CompilationMode.NORMAL,
        initial_layout: Any | None = None,
        resource_policy: Any | None = None,
        seed: int | None = None,
        qubit_order: cirq.QubitOrderOrList | None = None,
    ) -> None:
        self._connector = connector
        self._normalized_device = device
        self._device = TianyanDevice(device, connector=connector)
        self._compiler = compiler or CircuitCompiler()
        self._timeout = float(timeout)
        self._poll_interval = float(poll_interval)
        if self._timeout <= 0 or self._poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        self._calibration = CalibrationMode(str(calibration).strip().lower())
        self._require_available = bool(require_available)
        self._compilation_options = CompilationOptions(
            mode=CompilationMode(str(compilation_mode).strip().lower()),
            initial_layout=initial_layout,
            resource_policy=resource_policy,
            seed=seed,
        )
        self._qubit_order = qubit_order
        self._last_executions: tuple[CirqExecution, ...] = ()

    @property
    def device(self) -> TianyanDevice:
        return self._device

    @property
    def normalized_device(self) -> NormalizedDevice:
        return self._normalized_device

    @property
    def last_executions(self) -> tuple[CirqExecution, ...]:
        return self._last_executions

    @property
    def last_task_ids(self) -> tuple[str, ...]:
        return tuple(task_id for item in self._last_executions for task_id in item.task_ids)

    @property
    def last_qcis(self) -> tuple[str, ...]:
        return tuple(item.qcis for item in self._last_executions)

    def refresh_device(self) -> NormalizedDevice:
        """Refresh mutable Tianyan status fields."""

        self._normalized_device = self._device.refresh()
        return self._normalized_device

    def is_available(self, *, refresh: bool = True) -> bool:
        if refresh:
            self.refresh_device()
        return self._normalized_device.available

    @classmethod
    def from_connector(
        cls,
        connector: TianyanConnector,
        device_name: str,
        **kwargs: Any,
    ) -> TianyanSampler:
        return cls(connector, connector.resolve_device(device_name), **kwargs)

    @classmethod
    def login(
        cls,
        api_key: str,
        device_name: str,
        *,
        save_credentials: bool = False,
        login_options: Mapping[str, Any] | None = None,
        **sampler_options: Any,
    ) -> TianyanSampler:
        connector = TianyanConnector.login(
            api_key,
            save_credentials=save_credentials,
            **dict(login_options or {}),
        )
        return cls.from_connector(connector, device_name, **sampler_options)

    @classmethod
    def from_credentials(
        cls,
        device_name: str,
        *,
        credential_options: Mapping[str, Any] | None = None,
        **sampler_options: Any,
    ) -> TianyanSampler:
        connector = TianyanConnector.from_credentials(**dict(credential_options or {}))
        return cls.from_connector(connector, device_name, **sampler_options)

    def submit_sweep(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> tuple[CirqExecution, ...]:
        """Resolve and submit a sweep without waiting for cloud results."""

        if not isinstance(program, cirq.AbstractCircuit):
            raise TypeError("program must be a cirq.AbstractCircuit")
        if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
            raise ValueError("repetitions must be a positive integer")
        resolvers = tuple(cirq.to_resolvers(params))
        executions: list[CirqExecution] = []
        self._last_executions = ()
        for index, resolver in enumerate(resolvers):
            resolved = cirq.resolve_parameters(program, resolver, recursive=True)
            remaining = sorted(cirq.parameter_names(resolved))
            if remaining:
                raise ValueError(f"unresolved Cirq parameters after sweep resolution: {remaining}")
            bundle = cirq_to_cqlib(resolved, qubit_order=self._qubit_order)
            artifact = self._compiler.compile(
                bundle,
                device=self._normalized_device,
                options=self._compilation_options,
                circuit_index=index,
            )
            run_options = RunOptions(
                device_name=self._normalized_device.name,
                shots=repetitions,
                timeout=self._timeout,
                poll_interval=self._poll_interval,
                calibration=self._calibration,
                require_available=self._require_available,
            )
            adapter_job = self._connector.submit((artifact,), options=run_options)
            executions.append(CirqExecution(adapter_job, artifact, resolver))
            self._last_executions = tuple(executions)
        return tuple(executions)

    def submit(
        self,
        program: cirq.AbstractCircuit,
        *,
        param_resolver: cirq.ParamResolver | Mapping[str, Any] | None = None,
        repetitions: int = 1,
    ) -> CirqExecution:
        """Submit one resolved circuit and immediately return its task."""

        resolver = (
            param_resolver
            if isinstance(param_resolver, cirq.ParamResolver)
            else (cirq.ParamResolver(param_resolver or {}))
        )
        executions = self.submit_sweep(
            program,
            resolver,
            repetitions=repetitions,
        )
        if len(executions) != 1:
            raise RuntimeError("one Cirq submit call must create exactly one execution")
        return executions[0]

    def run_sweep(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> Sequence[cirq.Result]:
        """Submit a sweep, wait, and return standard Cirq results."""

        return [
            execution.result(
                timeout=self._timeout,
                poll_interval=self._poll_interval,
            )
            for execution in self.submit_sweep(program, params, repetitions)
        ]


__all__ = ["TianyanSampler"]
