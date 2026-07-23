from __future__ import annotations

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("cqlib")
from cqlib.device import Device, Topology
from qiskit import QuantumCircuit, transpile

from cqlib_adapter.common import (
    AdapterDeviceError,
    Coupling,
    DeviceStatus,
    DeviceToll,
    NormalizedDevice,
)
from cqlib_adapter.qiskit import coupling_map_from_device, target_from_device
from cqlib_adapter.qiskit.testing import MockCloudBackend

pytestmark = pytest.mark.qiskit


def normalized_line(size: int = 3) -> NormalizedDevice:
    return NormalizedDevice.from_backend(MockCloudBackend([], size=size))


def test_target_exposes_native_gates_measurements_and_topology() -> None:
    device = normalized_line()

    target = target_from_device(device)
    coupling_map = coupling_map_from_device(device)

    assert target.num_qubits == 3
    assert {"rz", "x2p", "x2m", "y2p", "y2m", "cz", "measure", "barrier"} <= set(
        target.operation_names
    )
    assert {tuple(edge) for edge in coupling_map.get_edges()} == {(0, 1), (1, 2)}
    assert set(target["measure"]) == {(0,), (1,), (2,)}


def test_target_drives_qiskit_gate_decomposition_and_routing() -> None:
    target = target_from_device(normalized_line())
    circuit = QuantumCircuit(3, 2)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.measure(0, 0)
    circuit.measure(2, 1)

    transpiled = transpile(
        circuit,
        target=target,
        optimization_level=1,
        seed_transpiler=7,
    )

    assert set(transpiled.count_ops()) <= set(target.operation_names)
    assert "cz" in transpiled.count_ops()
    for item in transpiled.data:
        if item.operation.num_qubits == 2:
            edge = tuple(transpiled.find_bit(qubit).index for qubit in item.qubits)
            assert edge in target["cz"]


def test_sparse_physical_ids_are_rejected_at_qiskit_boundary() -> None:
    config = Device("sparse", [1, 3], Topology([1, 3], [(1, 3, "CZ")]))
    device = NormalizedDevice(
        name="sparse",
        display_name="Sparse",
        num_qubits=2,
        qubits=(1, 3),
        native_gates=("RZ",),
        couplings=(),
        usable_qubits=(1, 3),
        invalid_qubits=(),
        status=DeviceStatus.RUNNING,
        toll=DeviceToll.FREE,
        available=True,
        cqlib_device=config,
    )

    with pytest.raises(AdapterDeviceError, match="zero-based dense"):
        target_from_device(device)


def test_unknown_device_native_gate_is_not_silently_dropped() -> None:
    config = Device.line("unknown", 1)
    device = NormalizedDevice(
        name="unknown",
        display_name="Unknown gate device",
        num_qubits=1,
        qubits=(0,),
        native_gates=("NOT_A_GATE",),
        couplings=(),
        usable_qubits=(0,),
        invalid_qubits=(),
        status=DeviceStatus.RUNNING,
        toll=DeviceToll.FREE,
        available=True,
        cqlib_device=config,
    )

    with pytest.raises(AdapterDeviceError, match="unsupported native gate"):
        target_from_device(device)


def test_coupling_dataclass_remains_compatible_with_target_edges() -> None:
    device = normalized_line(2)
    assert Coupling(0, 1) in device.couplings


def test_target_maps_qcis_directives_and_aliases() -> None:
    config = Device.line("directives", 1)
    device = NormalizedDevice(
        name="directives",
        display_name="Directive device",
        num_qubits=1,
        qubits=(0,),
        native_gates=("ID", "RESET", "DELAY", "M", "B", "GPHASE"),
        couplings=(),
        usable_qubits=(0,),
        invalid_qubits=(),
        status=DeviceStatus.RUNNING,
        toll=DeviceToll.FREE,
        available=True,
        cqlib_device=config,
    )

    target = target_from_device(device)

    assert {"id", "reset", "delay", "measure", "barrier"} <= set(target.operation_names)
    assert "gphase" not in target.operation_names
