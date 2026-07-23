"""Real Tianyan topology/layout smoke test for the Cirq adapter.

This script reads the API key from the current shell and reuses the device
selector from 03_tianyan_cloud.py. It selects a
connected three-qubit physical path, maps Cirq q0/q1/q2 onto it, validates the
compiled cqlib operations and measurements locally, and only then submits.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from time import perf_counter
from typing import Any

import cirq
from cqlib.device import Layout

from cqlib_adapter.cirq import TianyanSampler, compile_cirq_circuit
from cqlib_adapter.common import CompilationOptions, NormalizedDevice, TianyanConnector

SEED = 43
IDEAL_OUTCOMES = {4, 7}  # Cirq big-endian measurement integers: 100 and 111.
MIN_SUCCESS_PROBABILITY = 0.5


def load_cloud_config() -> tuple[str, str]:
    source = runpy.run_path(str(Path(__file__).with_name("03_tianyan_cloud.py")))
    api_key = os.getenv("TIANYAN_API_KEY", "").strip()
    device_name = str(source["DEFAULT_DEVICE"]).strip()
    if not api_key:
        raise RuntimeError("set TIANYAN_API_KEY in the current shell first")
    if not device_name:
        raise RuntimeError("fill DEFAULT_DEVICE in 03_tianyan_cloud.py first")
    return api_key, device_name


def select_three_qubit_path(device: NormalizedDevice) -> tuple[int, int, int]:
    usable = set(device.usable_qubits)
    adjacency = {qubit: set() for qubit in usable}
    for edge in device.couplings:
        if edge.source in usable and edge.target in usable:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
    for center in sorted(adjacency):
        neighbors = sorted(adjacency[center])
        if len(neighbors) >= 2:
            return neighbors[0], center, neighbors[1]
    raise RuntimeError("device has no connected path containing three usable qubits")


def topology_circuit() -> cirq.Circuit:
    q0, q1, q2 = cirq.LineQubit.range(3)
    return cirq.Circuit(
        cirq.X(q0),
        cirq.H(q1),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q1, q2),
        cirq.measure(q0, q1, q2, key="state"),
    )


def operation_qubits(operation: Any) -> tuple[int, ...]:
    return tuple(qubit.index for qubit in operation.qubits)


def assert_compiled_mapping(
    artifact: Any,
    device: NormalizedDevice,
    physical_path: tuple[int, int, int],
) -> None:
    left, center, right = physical_path
    selected = set(physical_path)
    expected_edges = {frozenset((left, center)), frozenset((center, right))}
    device_edges = {
        frozenset((edge.source, edge.target))
        for edge in device.couplings
        if edge.source in device.usable_qubits and edge.target in device.usable_qubits
    }
    if not expected_edges <= device_edges:
        raise AssertionError("selected physical path is absent from cqlib device topology")

    compiled_qubits: set[int] = set()
    two_qubit_edges: list[frozenset[int]] = []
    for operation in artifact.circuit.operations:
        qubits = operation_qubits(operation)
        compiled_qubits.update(qubits)
        if len(qubits) == 2:
            two_qubit_edges.append(frozenset(qubits))
    if not compiled_qubits <= selected:
        raise AssertionError(
            f"compiler escaped selected path: used={sorted(compiled_qubits)}, "
            f"selected={sorted(selected)}"
        )
    if not two_qubit_edges:
        raise AssertionError("compiled circuit contains no two-qubit operation")
    if not set(two_qubit_edges) <= expected_edges:
        raise AssertionError(f"compiled two-qubit gates violate topology: {two_qubit_edges}")

    measured = tuple(item.physical_qubit for item in artifact.measurements)
    if measured != physical_path:
        raise AssertionError(
            f"Cirq logical-to-physical measurement mapping changed: "
            f"expected={physical_path}, actual={measured}"
        )
    if tuple(item.key for item in artifact.measurements) != ("state",) * 3:
        raise AssertionError("Cirq measurement key was not preserved through compilation")


def main() -> None:
    api_key, device_selector = load_cloud_config()
    shots = int(os.getenv("TIANYAN_TEST_SHOTS", "100"))
    timeout = float(os.getenv("TIANYAN_TEST_TIMEOUT", "600"))
    poll_interval = float(os.getenv("TIANYAN_POLL_INTERVAL", "5"))
    if shots <= 0 or timeout <= 0 or poll_interval <= 0:
        raise ValueError("shots, timeout and poll interval must be positive")

    connector = TianyanConnector.login(api_key, save_credentials=False)
    cloud_device = connector.resolve_device(device_selector)
    physical_path = select_three_qubit_path(cloud_device)
    layout = Layout.from_pairs(
        [(logical, physical) for logical, physical in enumerate(physical_path)],
        physical_count=cloud_device.num_qubits,
    )
    circuit = topology_circuit()
    options = CompilationOptions(initial_layout=layout, seed=SEED)

    print(f"selected device: {cloud_device.display_name} [{cloud_device.name}]")
    print("device status:", cloud_device.status.value)
    print("selected physical path (Cirq q0, q1, q2):", physical_path)
    print("selected physical couplings:", physical_path[:2], physical_path[1:])
    print("Cirq circuit:")
    print(circuit)

    artifact = compile_cirq_circuit(circuit, device=cloud_device, options=options)
    assert_compiled_mapping(artifact, cloud_device, physical_path)
    print("compiled measurement mapping:", artifact.measurements)
    print("topology-validated QCIS:\n", artifact.qcis)
    print("PASS: Cirq qids were mapped onto the selected cqlib physical path.")

    if not cloud_device.available:
        print(
            f"SKIP: device {cloud_device.name!r} is currently "
            f"{cloud_device.status.value}; local topology validation passed, "
            "but no task was submitted."
        )
        return

    sampler = TianyanSampler(
        connector,
        cloud_device,
        timeout=timeout,
        poll_interval=poll_interval,
        calibration="auto",
        initial_layout=layout,
        seed=SEED,
    )
    started = perf_counter()
    result = sampler.run(circuit, repetitions=shots)
    elapsed = perf_counter() - started
    execution = sampler.last_executions[0]
    assert_compiled_mapping(execution.artifact, cloud_device, physical_path)
    if execution.qcis != artifact.qcis:
        raise AssertionError("submitted QCIS differs from prevalidated QCIS")

    histogram = dict(result.histogram(key="state"))
    print("task IDs:", execution.task_ids)
    print("final status:", execution.status())
    print("histogram (Cirq big-endian integers):", histogram)
    print("first ten measurement rows:\n", result.measurements["state"][:10])
    print(f"total compile + submit + wait + result time: {elapsed:.3f} s")

    if sum(histogram.values()) != shots:
        raise AssertionError(f"histogram total does not equal repetitions: {histogram}")
    success = sum(count for outcome, count in histogram.items() if outcome in IDEAL_OUTCOMES)
    success_probability = success / shots
    print("ideal Cirq outcomes: 4 (100), 7 (111)")
    print(f"ideal-outcome success probability: {success_probability:.3f}")
    if success_probability < MIN_SUCCESS_PROBABILITY:
        raise AssertionError(
            f"topology test probability {success_probability:.3f} is below "
            f"{MIN_SUCCESS_PROBABILITY:.3f}"
        )
    print("PASS: real Tianyan Cirq topology/layout execution completed successfully.")


if __name__ == "__main__":
    main()
