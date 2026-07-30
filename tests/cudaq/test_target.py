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
    Coupling,
    DeviceStatus,
    DeviceToll,
    NormalizedDevice,
)
from cqlib_adapter.cudaq import TianyanTarget, target_from_device

pytestmark = pytest.mark.cudaq


def device() -> NormalizedDevice:
    return NormalizedDevice(
        name="tianyan-test",
        display_name="Tianyan Test",
        num_qubits=3,
        qubits=(0, 1, 2),
        native_gates=("RZ", "CZ", "MEASURE"),
        couplings=(Coupling(0, 1), Coupling(1, 2)),
        usable_qubits=(0, 1, 2),
        invalid_qubits=(),
        status=DeviceStatus.RUNNING,
        toll=DeviceToll.FREE,
        available=True,
        cqlib_device=object(),  # type: ignore[arg-type]
    )


def test_target_exposes_tianyan_device_topology_and_native_gates() -> None:
    target = target_from_device(device())

    assert isinstance(target, TianyanTarget)
    assert target.name == "tianyan-test"
    assert target.display_name == "Tianyan Test"
    assert target.num_qubits == 3
    assert target.qubits == (0, 1, 2)
    assert target.native_gates == ("RZ", "CZ", "MEASURE")
    assert target.couplings == ((0, 1), (1, 2))
    assert target.supports("rz")
    assert target.supports_coupling(1, 0, either_direction=True)
    assert target.status == "running"
    assert target.toll == "free"
    assert target.available


def test_target_dictionary_is_read_only_and_refresh_without_connector_is_stable() -> None:
    target = target_from_device(device())
    values = target.as_dict()

    assert values["properties_complete"] is False
    assert target.refresh() is target.device
    try:
        values["name"] = "changed"  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("target dictionary must be read-only")
