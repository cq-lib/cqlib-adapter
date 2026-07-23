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

"""CUDA-Q kernel execution through cqlib compilation and Tianyan."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cqlib_adapter.common import (
    CalibrationMode,
    CircuitCompiler,
    CompilationMode,
    CompilationOptions,
    NormalizedDevice,
    RunOptions,
    TianyanConnector,
)

from .converter import cudaq_to_cqlib
from .job import CudaQJob
from .result import CudaQSampleResult
from .target import TianyanTarget, target_from_device


class TianyanExecutor:
    """Synchronous and asynchronous CUDA-Q sampling on Tianyan."""

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
    ) -> None:
        self._connector = connector
        self._device = device
        self._target = target_from_device(device, connector=connector)
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
        self._last_job: CudaQJob | None = None

    @classmethod
    def from_connector(
        cls,
        connector: TianyanConnector,
        device_name: str,
        **kwargs: Any,
    ) -> TianyanExecutor:
        return cls(connector, connector.resolve_device(device_name), **kwargs)

    @classmethod
    def login(
        cls,
        api_key: str,
        device_name: str,
        *,
        save_credentials: bool = False,
        login_options: Mapping[str, Any] | None = None,
        **executor_options: Any,
    ) -> TianyanExecutor:
        connector = TianyanConnector.login(
            api_key,
            save_credentials=save_credentials,
            **dict(login_options or {}),
        )
        return cls.from_connector(connector, device_name, **executor_options)

    @classmethod
    def from_credentials(
        cls,
        device_name: str,
        *,
        credential_options: Mapping[str, Any] | None = None,
        **executor_options: Any,
    ) -> TianyanExecutor:
        connector = TianyanConnector.from_credentials(**dict(credential_options or {}))
        return cls.from_connector(connector, device_name, **executor_options)

    @property
    def target(self) -> TianyanTarget:
        return self._target

    @property
    def device(self) -> NormalizedDevice:
        return self._device

    @property
    def last_job(self) -> CudaQJob | None:
        return self._last_job

    @property
    def last_task_id(self) -> str | None:
        return self._last_job.task_id if self._last_job is not None else None

    @property
    def last_qcis(self) -> str | None:
        return self._last_job.qcis if self._last_job is not None else None

    def refresh_device(self) -> NormalizedDevice:
        self._device = self._target.refresh()
        return self._device

    def is_available(self, *, refresh: bool = True) -> bool:
        if refresh:
            self.refresh_device()
        return self._device.available

    def submit(
        self,
        kernel: Any,
        *arguments: Any,
        shots_count: int = 1000,
    ) -> CudaQJob:
        """Compile and submit without waiting."""

        if isinstance(shots_count, bool) or not isinstance(shots_count, int) or shots_count <= 0:
            raise ValueError("shots_count must be a positive integer")
        bundle = cudaq_to_cqlib(kernel, *arguments)
        artifact = self._compiler.compile(
            bundle,
            device=self._device,
            options=self._compilation_options,
            circuit_index=0,
        )
        adapter_job = self._connector.submit(
            (artifact,),
            options=RunOptions(
                device_name=self._device.name,
                shots=shots_count,
                timeout=self._timeout,
                poll_interval=self._poll_interval,
                calibration=self._calibration,
                require_available=self._require_available,
            ),
        )
        self._last_job = CudaQJob(adapter_job, artifact)
        return self._last_job

    def sample_async(
        self,
        kernel: Any,
        *arguments: Any,
        shots_count: int = 1000,
    ) -> CudaQJob:
        """CUDA-Q-style asynchronous sampling entry point."""

        return self.submit(kernel, *arguments, shots_count=shots_count)

    def sample(
        self,
        kernel: Any,
        *arguments: Any,
        shots_count: int = 1000,
    ) -> CudaQSampleResult:
        """Compile, submit, wait and return CUDA-Q-compatible counts."""

        return self.submit(kernel, *arguments, shots_count=shots_count).result(
            timeout=self._timeout,
            poll_interval=self._poll_interval,
        )

    def run(
        self,
        kernel: Any,
        *arguments: Any,
        shots_count: int = 1000,
    ) -> CudaQSampleResult:
        return self.sample(kernel, *arguments, shots_count=shots_count)


__all__ = ["TianyanExecutor"]
