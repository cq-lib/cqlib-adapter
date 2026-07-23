"""Qiskit BackendV2 surface backed by cqlib and Tianyan."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

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
)

from .converter import qiskit_to_cqlib
from .job import TianyanJob
from .target import target_from_device


@dataclass(frozen=True, slots=True)
class TianyanBackendStatus:
    """Small Qiskit-style snapshot for versions where BackendStatus was removed."""

    backend_name: str
    backend_version: str
    operational: bool
    pending_jobs: int
    status_msg: str


class TianyanBackend(BackendV2):
    """Compile Qiskit circuits with cqlib and execute them on Tianyan."""

    _SUPPORTED_RUN_OPTIONS = frozenset(
        {
            "shots",
            "timeout",
            "poll_interval",
            "calibration",
            "require_available",
            "compilation_mode",
            "seed",
            "seed_simulator",
            "initial_layout",
            "resource_policy",
            "memory",
        }
    )

    def __init__(
        self,
        connector: TianyanConnector,
        device: NormalizedDevice,
        *,
        compiler: CircuitCompiler | None = None,
        max_circuits: int | None = 50,
    ) -> None:
        if max_circuits is not None and max_circuits <= 0:
            raise ValueError("max_circuits must be positive or None")
        self._connector = connector
        self._device = device
        self._compiler = compiler or CircuitCompiler()
        self._target = target_from_device(device)
        self._max_circuits = max_circuits
        super().__init__(
            name=device.name,
            description=device.display_name,
            backend_version="2.0.0",
        )

    @classmethod
    def from_connector(
        cls,
        connector: TianyanConnector,
        device_name: str,
        **kwargs: Any,
    ) -> TianyanBackend:
        """Create a BackendV2 from an authenticated or mocked connector."""

        return cls(connector, connector.get_device(device_name), **kwargs)

    @classmethod
    def login(
        cls,
        api_key: str,
        device_name: str,
        **login_options: Any,
    ) -> TianyanBackend:
        """Authenticate with cqlib-tianyan and create one BackendV2."""

        connector = TianyanConnector.login(api_key, **login_options)
        return cls.from_connector(connector, device_name)

    @classmethod
    def from_credentials(
        cls,
        device_name: str,
        **credential_options: Any,
    ) -> TianyanBackend:
        """Load saved cqlib-tianyan credentials and create one BackendV2."""

        connector = TianyanConnector.from_credentials(**credential_options)
        return cls.from_connector(connector, device_name)

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024,
            timeout=120.0,
            poll_interval=5.0,
            calibration=CalibrationMode.AUTO.value,
            require_available=True,
            compilation_mode=CompilationMode.NORMAL.value,
            seed=None,
            seed_simulator=None,
            initial_layout=None,
            resource_policy=None,
            memory=True,
        )

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> int | None:
        return self._max_circuits

    @property
    def device(self) -> NormalizedDevice:
        """Normalized cqlib-tianyan device information."""

        return self._device

    @property
    def device_status(self) -> DeviceStatus:
        return self._device.status

    def refresh_device(self) -> NormalizedDevice:
        """Refresh mutable Tianyan status fields."""

        self._device = self._connector.refresh_device_state(self._device)
        return self._device

    def is_available(self, *, refresh: bool = True) -> bool:
        if refresh:
            self.refresh_device()
        return self._device.available

    def status(self) -> TianyanBackendStatus:
        device = self.refresh_device()
        return TianyanBackendStatus(
            backend_name=self.name,
            backend_version=self.backend_version,
            operational=device.available,
            pending_jobs=0,
            status_msg=device.status.value,
        )

    def _option(self, name: str, overrides: dict[str, Any]) -> Any:
        return overrides.get(name, getattr(self.options, name))

    @staticmethod
    def _circuits(run_input: Any) -> tuple[QuantumCircuit, ...]:
        if isinstance(run_input, QuantumCircuit):
            return (run_input,)
        if isinstance(run_input, Sequence) and not isinstance(run_input, str | bytes | bytearray):
            circuits = tuple(run_input)
            if not circuits:
                raise ValueError("run_input must contain at least one circuit")
            if not all(isinstance(circuit, QuantumCircuit) for circuit in circuits):
                raise TypeError("every run_input item must be a qiskit.QuantumCircuit")
            return circuits
        raise TypeError("run_input must be a qiskit.QuantumCircuit or a sequence of circuits")

    def run(self, run_input: Any, **options: Any) -> TianyanJob:
        """Compile, submit and return an already-submitted Qiskit job."""

        unknown = sorted(set(options) - self._SUPPORTED_RUN_OPTIONS)
        if unknown:
            raise TypeError(f"unsupported Tianyan backend run options: {unknown}")
        circuits = self._circuits(run_input)
        if self.max_circuits is not None and len(circuits) > self.max_circuits:
            raise ValueError(f"backend accepts at most {self.max_circuits} circuits per run")

        shots = self._option("shots", options)
        timeout = self._option("timeout", options)
        poll_interval = self._option("poll_interval", options)
        calibration = CalibrationMode(str(self._option("calibration", options)).strip().lower())
        compilation_mode = CompilationMode(
            str(self._option("compilation_mode", options)).strip().lower()
        )
        run_options = RunOptions(
            device_name=self.name,
            shots=shots,
            timeout=timeout,
            poll_interval=poll_interval,
            calibration=calibration,
            require_available=bool(self._option("require_available", options)),
        )
        seed = self._option("seed", options)
        if seed is None:
            seed = self._option("seed_simulator", options)
        compilation_options = CompilationOptions(
            mode=compilation_mode,
            initial_layout=self._option("initial_layout", options),
            resource_policy=self._option("resource_policy", options),
            seed=seed,
        )

        bundles = tuple(qiskit_to_cqlib(circuit) for circuit in circuits)
        for index, bundle in enumerate(bundles):
            if not bundle.metadata.measurements.slots:
                raise AdapterConversionError(f"Qiskit circuit at index {index} has no measurements")
        artifacts = tuple(
            self._compiler.compile(
                bundle,
                device=self._device,
                options=compilation_options,
                circuit_index=index,
            )
            for index, bundle in enumerate(bundles)
        )
        adapter_job = self._connector.submit(artifacts, options=run_options)
        return TianyanJob(self, adapter_job, artifacts)


__all__ = ["TianyanBackend", "TianyanBackendStatus"]
