"""Tianyan connection, discovery, compilation, and submission core."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from importlib import import_module
from typing import Any, cast

from .circuit import TranslationBundle
from .compiler import CircuitCompiler, CompilationArtifact
from .device import DeviceStatus, DeviceToll, NormalizedDevice
from .errors import (
    AdapterDeviceError,
    AdapterSubmissionError,
    ErrorContext,
)
from .job import AdapterJob
from .options import CalibrationMode, CompilationOptions, RunOptions
from .typing import (
    BackendLike,
    CircuitLike,
    PlatformLike,
    TaskHandleLike,
)


class TianyanConnector:
    """Single cloud boundary reused by every framework-specific surface."""

    def __init__(
        self,
        platform: PlatformLike,
        *,
        compiler: CircuitCompiler | None = None,
    ) -> None:
        self._platform = platform
        self._compiler = compiler or CircuitCompiler()

    @classmethod
    def login(
        cls,
        api_key: str,
        **kwargs: Any,
    ) -> TianyanConnector:
        secret = api_key.strip()
        if not secret:
            raise ValueError("api_key must not be empty")
        try:
            platform_type = import_module("cqlib_tianyan").TianyanPlatform
            platform = platform_type.login(secret, **kwargs)
            return cls(cast(PlatformLike, platform))
        except Exception as exc:
            detail = str(exc).replace(secret, "<redacted>")
        suffix = f": {detail}" if detail else ""
        raise AdapterSubmissionError(f"Tianyan login failed{suffix}") from None

    @classmethod
    def from_credentials(cls, **kwargs: Any) -> TianyanConnector:
        try:
            platform_type = import_module("cqlib_tianyan").TianyanPlatform
            platform = platform_type.from_credentials(**kwargs)
            return cls(cast(PlatformLike, platform))
        except Exception:
            # Providers may include token material in their exception text, so
            # do not forward the underlying message or exception chain.
            credential_error = "failed to load Tianyan credentials"
        raise AdapterSubmissionError(credential_error) from None

    def list_devices(self) -> tuple[NormalizedDevice, ...]:
        try:
            return tuple(
                NormalizedDevice.from_backend(backend) for backend in self._platform.list_backends()
            )
        except AdapterDeviceError:
            raise
        except Exception as exc:
            raise AdapterDeviceError(f"failed to list Tianyan devices: {exc}") from exc

    def resolve_device(self, selector: str) -> NormalizedDevice:
        """Resolve an exact backend name/display name, then download only its config."""

        normalized = selector.strip().casefold()
        if not normalized:
            raise ValueError("device selector must not be empty")
        try:
            backends = tuple(self._platform.list_backends())
            matches = [
                backend
                for backend in backends
                if normalized
                in {
                    str(backend.name).strip().casefold(),
                    str(backend.display_name or "").strip().casefold(),
                }
            ]
            if not matches:
                available = ", ".join(
                    f"{backend.display_name or backend.name} [{backend.name}]"
                    for backend in backends
                )
                raise AdapterDeviceError(
                    f"Tianyan device {selector!r} was not found; available: {available}"
                )
            if len(matches) > 1:
                names = ", ".join(str(backend.name) for backend in matches)
                raise AdapterDeviceError(
                    f"Tianyan device selector {selector!r} is ambiguous: {names}"
                )
            return NormalizedDevice.from_backend(matches[0])
        except AdapterDeviceError:
            raise
        except Exception as exc:
            raise AdapterDeviceError(
                f"failed to resolve Tianyan device {selector!r}: {exc}",
                context=ErrorContext(device_name=selector.strip()),
            ) from exc

    def get_device(self, name: str) -> NormalizedDevice:
        if not name.strip():
            raise ValueError("device name must not be empty")
        try:
            return NormalizedDevice.from_backend(self._platform.get_backend(name.strip()))
        except AdapterDeviceError:
            raise
        except Exception as exc:
            raise AdapterDeviceError(
                f"failed to get Tianyan device: {exc}",
                context=ErrorContext(device_name=name.strip()),
            ) from exc

    def compile_and_submit(
        self,
        bundles: Sequence[TranslationBundle[CircuitLike]],
        *,
        run_options: RunOptions,
        compilation_options: CompilationOptions | None = None,
    ) -> AdapterJob:
        if not bundles:
            raise ValueError("at least one circuit is required")
        device = self.get_device(run_options.device_name)
        artifacts = tuple(
            self._compiler.compile(
                bundle,
                device=device,
                options=compilation_options,
                circuit_index=index,
            )
            for index, bundle in enumerate(bundles)
        )
        return self.submit(artifacts, options=run_options)

    def refresh_device_state(self, device: NormalizedDevice) -> NormalizedDevice:
        try:
            backend = self._platform.get_backend(device.name)
            return replace(
                device,
                display_name=backend.display_name or backend.name,
                status=DeviceStatus.parse(backend.status),
                toll=DeviceToll.parse(backend.toll),
                available=bool(backend.is_available()),
            )
        except Exception as exc:
            raise AdapterDeviceError(
                f"failed to refresh Tianyan device state: {exc}",
                context=ErrorContext(device_name=device.name),
            ) from exc

    def submit(
        self,
        artifacts: Sequence[CompilationArtifact],
        *,
        options: RunOptions,
    ) -> AdapterJob:
        if not artifacts:
            raise ValueError("at least one compilation artifact is required")
        try:
            backend = self._platform.get_backend(options.device_name)
            if options.require_available and not backend.is_available():
                raise AdapterDeviceError(
                    "Tianyan device is not available",
                    context=ErrorContext(device_name=options.device_name),
                )
        except AdapterDeviceError:
            raise
        except Exception as exc:
            raise AdapterDeviceError(
                f"failed to prepare Tianyan device: {exc}",
                context=ErrorContext(device_name=options.device_name),
            ) from exc

        for index, artifact in enumerate(artifacts):
            if artifact.device is None or artifact.device.name != options.device_name:
                raise AdapterSubmissionError(
                    "artifact was not compiled for the requested device",
                    context=ErrorContext(
                        device_name=options.device_name,
                        circuit_index=index,
                    ),
                )

        handles: list[TaskHandleLike] = []
        for index, artifact in enumerate(artifacts):
            try:
                handles.append(self._submit_one(backend, artifact, options))
            except Exception as exc:
                submitted = [task_id for handle in handles for task_id in handle.task_ids]
                suffix = f"; already submitted task IDs: {submitted}" if submitted else ""
                raise AdapterSubmissionError(
                    f"Tianyan submission failed: {exc}{suffix}",
                    context=ErrorContext(
                        device_name=options.device_name,
                        circuit_index=index,
                    ),
                ) from exc
        return AdapterJob(
            handles,
            artifacts,
            default_timeout=options.timeout,
            default_poll_interval=options.poll_interval,
        )

    @staticmethod
    def _submit_one(
        backend: BackendLike,
        artifact: CompilationArtifact,
        options: RunOptions,
    ) -> TaskHandleLike:
        circuits = [artifact.qcis]
        if options.calibration is CalibrationMode.DISABLED:
            return backend.run_raw(circuits, options.shots)
        if options.calibration is CalibrationMode.ENABLED:
            return backend.run_with_mode(circuits, options.shots, "enabled")
        return backend.run(circuits, options.shots)


__all__ = ["TianyanConnector"]
