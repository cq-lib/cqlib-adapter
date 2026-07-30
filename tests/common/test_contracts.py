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

from __future__ import annotations

import pytest

from cqlib_adapter.common import (
    AdapterConversionError,
    AdapterDeviceError,
    CompilationMode,
    CompilationOptions,
    Coupling,
    DeviceStatus,
    DeviceToll,
    ErrorContext,
    MeasurementMetadata,
    MeasurementSlot,
    NormalizedDevice,
    QubitMapping,
    RunOptions,
    TranslationMetadata,
)

from .fakes import FakeBackend, FakeDeviceConfig

pytestmark = pytest.mark.unit


def test_error_context_is_rendered() -> None:
    error = AdapterDeviceError(
        "offline",
        context=ErrorContext(device_name="ty", task_id="42", circuit_index=1),
    )
    assert str(error) == "device: offline (device=ty, task_id=42, circuit_index=1)"


def test_qubit_mapping_is_bijective_and_immutable() -> None:
    source = {"q0": 2, "q1": 5}
    mapping = QubitMapping(source)
    source["q0"] = 9
    assert mapping.physical("q0") == 2
    assert mapping.physical_to_logical == {2: "q0", 5: "q1"}
    with pytest.raises(TypeError):
        mapping.logical_to_physical["q2"] = 7  # type: ignore[index]


@pytest.mark.parametrize(
    "mapping",
    [{"a": -1}, {"a": 0, "b": 0}, {"": 1}, {"a": True}],
)
def test_qubit_mapping_rejects_invalid_values(mapping: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=r"qubit|physical|bijective"):
        QubitMapping(mapping)


def test_unknown_logical_qubit_uses_conversion_error() -> None:
    with pytest.raises(AdapterConversionError, match="unknown logical qubit"):
        QubitMapping({"q": 0}).physical("missing")


def test_measurement_metadata_validates_classical_layout() -> None:
    metadata = MeasurementMetadata(
        slots=(MeasurementSlot("q0", 1), MeasurementSlot("q1", 0)),
        num_classical_bits=2,
        register_sizes={"c": 2},
    )
    assert metadata.num_classical_bits == 2
    with pytest.raises(ValueError, match="at most one"):
        MeasurementMetadata(
            slots=(MeasurementSlot("q0", 0), MeasurementSlot("q1", 0)),
            num_classical_bits=1,
        )
    with pytest.raises(ValueError, match="sum"):
        MeasurementMetadata(num_classical_bits=2, register_sizes={"c": 1})


def test_translation_metadata_rejects_unknown_measured_qubit() -> None:
    measurements = MeasurementMetadata((MeasurementSlot("missing", 0),), 1)
    with pytest.raises(ValueError, match="unknown logical"):
        TranslationMetadata("qiskit", ("q0",), measurements)


def test_compilation_options_normalize_basis() -> None:
    options = CompilationOptions(
        mode=CompilationMode.ENHANCED,
        target_basis=(" x90 ", "cz"),
        seed=0,
    )
    assert options.target_basis == ("X90", "CZ")
    with pytest.raises(ValueError, match="duplicate"):
        CompilationOptions(target_basis=("cz", "CZ"))
    with pytest.raises(ValueError, match="seed"):
        CompilationOptions(seed=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"device_name": " "}, "device_name"),
        ({"device_name": "d", "shots": 0}, "shots"),
        ({"device_name": "d", "timeout": 0}, "timeout"),
        ({"device_name": "d", "poll_interval": 0}, "poll_interval"),
    ],
)
def test_run_options_reject_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RunOptions(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("timeout", float("nan"), ValueError),
        ("timeout", float("inf"), ValueError),
        ("timeout", float("-inf"), ValueError),
        ("timeout", True, TypeError),
        ("timeout", "5", TypeError),
        ("poll_interval", float("nan"), ValueError),
        ("poll_interval", float("inf"), ValueError),
        ("poll_interval", float("-inf"), ValueError),
        ("poll_interval", True, TypeError),
        ("poll_interval", "5", TypeError),
    ],
)
def test_run_options_reject_non_finite_or_non_numeric_wait_values(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match=field):
        RunOptions("d", **{field: value})  # type: ignore[arg-type]


def test_device_normalization_exposes_topology_and_state() -> None:
    config = FakeDeviceConfig(
        size=3,
        gates=("x90", "CZ", "cz"),
        edges=((0, 1), (1, 0), (1, 2)),
        invalid=(2,),
    )
    device = NormalizedDevice.from_backend(FakeBackend(config=config))
    assert device.name == "fake"
    assert device.display_name == "fake display"
    assert device.num_qubits == 3
    assert device.native_gates == ("X90", "CZ")
    assert device.usable_qubits == (0, 1)
    assert device.invalid_qubits == (2,)
    assert device.qubits == (0, 1, 2)
    assert not device.properties_complete
    assert device.status is DeviceStatus.RUNNING
    assert device.toll is DeviceToll.FREE
    assert device.supports("cz")
    assert device.supports_coupling(1, 0)
    assert device.supports_coupling(2, 1, either_direction=True)


def test_device_enum_parsing_has_safe_unknown() -> None:
    assert DeviceStatus.parse("running-now") is DeviceStatus.UNKNOWN
    assert DeviceStatus.parse("available") is DeviceStatus.RUNNING
    assert DeviceStatus.parse("calibration") is DeviceStatus.CALIBRATION
    assert DeviceStatus.parse("under_maintenance") is DeviceStatus.UNDER_MAINTENANCE
    assert DeviceToll.parse("charged") is DeviceToll.PAID


def test_device_normalization_wraps_malformed_config() -> None:
    config = FakeDeviceConfig(size=2, edges=((0, 3),))
    with pytest.raises(AdapterDeviceError, match="normalize"):
        NormalizedDevice.from_backend(FakeBackend(config=config))


def test_normalized_device_requires_qubit_partition() -> None:
    config = FakeDeviceConfig(size=2)
    with pytest.raises(ValueError, match="partition"):
        NormalizedDevice(
            name="bad",
            display_name="bad",
            num_qubits=2,
            qubits=(0, 1),
            native_gates=("X",),
            couplings=(Coupling(0, 1),),
            usable_qubits=(0,),
            invalid_qubits=(),
            status=DeviceStatus.RUNNING,
            toll=DeviceToll.FREE,
            available=True,
            cqlib_device=config,
        )


def test_normalized_device_supports_non_contiguous_physical_ids() -> None:
    config = FakeDeviceConfig(size=2)
    device = NormalizedDevice(
        name="sparse",
        display_name="sparse",
        num_qubits=2,
        qubits=(1, 8),
        native_gates=("CZ",),
        couplings=(Coupling(1, 8),),
        usable_qubits=(1, 8),
        invalid_qubits=(),
        status=DeviceStatus.RUNNING,
        toll=DeviceToll.FREE,
        available=True,
        cqlib_device=config,
    )
    assert device.qubits == (1, 8)
    assert device.supports_coupling(8, 1, either_direction=True)
