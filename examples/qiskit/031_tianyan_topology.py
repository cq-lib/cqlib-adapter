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

"""Real Tianyan topology/layout smoke test for the Qiskit adapter.

The script reads the API key from the current shell and reuses the device
selector configured in 03_tianyan_cloud.py.
It first verifies Qiskit Target <- cqlib topology and compiles a three-qubit
logical circuit onto one connected physical path. A real task is submitted
only after every local topology/layout assertion passes and the device is
available.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from time import perf_counter
from typing import Any

from cqlib.device import Layout
from qiskit import QuantumCircuit

from cqlib_adapter.common import CompilationOptions, NormalizedDevice, TianyanConnector
from cqlib_adapter.qiskit import TianyanBackend, compile_qiskit_circuit

SEED = 37
IDEAL_OUTCOMES = {"001", "111"}
MIN_SUCCESS_PROBABILITY = 0.5


def load_cloud_config() -> tuple[str, str]:
    """Read the shell-only secret and reuse the device selector from example 03."""

    source = runpy.run_path(str(Path(__file__).with_name("03_tianyan_cloud.py")))
    api_key = os.getenv("TIANYAN_API_KEY", "").strip()
    device_name = str(source["DEFAULT_DEVICE"]).strip()
    if not api_key:
        raise RuntimeError("set TIANYAN_API_KEY in the current shell first")
    if not device_name:
        raise RuntimeError("fill DEFAULT_DEVICE in 03_tianyan_cloud.py first")
    return api_key, device_name


def select_three_qubit_path(device: NormalizedDevice) -> tuple[int, int, int]:
    """Select deterministic left-center-right usable physical qubits."""

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


def operation_qubits(operation: Any) -> tuple[int, ...]:
    return tuple(qubit.index for qubit in operation.qubits)


def assert_compiled_mapping(
    artifact: Any,
    backend: TianyanBackend,
    physical_path: tuple[int, int, int],
) -> None:
    """Verify Target edges, compiled operations and measurements agree."""

    left, center, right = physical_path
    selected = set(physical_path)
    expected_edges = {frozenset((left, center)), frozenset((center, right))}
    target_edges = {frozenset(edge) for edge in backend.target["cz"] if len(edge) == 2}
    if not expected_edges <= target_edges:
        raise AssertionError(
            f"Qiskit Target omits selected cqlib topology edges: "
            f"expected={expected_edges}, target={target_edges}"
        )

    compiled_qubits: set[int] = set()
    compiled_two_qubit_edges: list[frozenset[int]] = []
    for operation in artifact.circuit.operations:
        qubits = operation_qubits(operation)
        compiled_qubits.update(qubits)
        if len(qubits) == 2:
            compiled_two_qubit_edges.append(frozenset(qubits))
    if not compiled_qubits <= selected:
        raise AssertionError(
            f"compiler escaped the requested physical path: "
            f"used={sorted(compiled_qubits)}, selected={sorted(selected)}"
        )
    if not compiled_two_qubit_edges:
        raise AssertionError("compiled circuit contains no two-qubit operation")
    if not set(compiled_two_qubit_edges) <= expected_edges:
        raise AssertionError(
            f"compiled two-qubit operation violates selected topology path: "
            f"actual={compiled_two_qubit_edges}, expected={expected_edges}"
        )

    measured = tuple(item.physical_qubit for item in artifact.measurements)
    if measured != physical_path:
        raise AssertionError(
            f"logical-to-physical measurement mapping changed: "
            f"expected={physical_path}, actual={measured}"
        )


def topology_circuit() -> QuantumCircuit:
    """Prepare q0=1 and a Bell-correlated q1/q2 pair, then measure all."""

    circuit = QuantumCircuit(3, 3, name="qiskit-live-topology-layout")
    circuit.x(0)
    circuit.h(1)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure([0, 1, 2], [0, 1, 2])
    return circuit


def main() -> None:
    api_key, device_selector = load_cloud_config()
    shots = int(os.getenv("TIANYAN_TEST_SHOTS", "100"))
    timeout = float(os.getenv("TIANYAN_TEST_TIMEOUT", "600"))
    poll_interval = float(os.getenv("TIANYAN_POLL_INTERVAL", "5"))
    if shots <= 0 or timeout <= 0 or poll_interval <= 0:
        raise ValueError("shots, timeout and poll interval must be positive")

    connector = TianyanConnector.login(api_key, save_credentials=False)
    device = connector.resolve_device(device_selector)
    backend = TianyanBackend(connector, device)
    physical_path = select_three_qubit_path(device)
    layout = Layout.from_pairs(
        [(logical, physical) for logical, physical in enumerate(physical_path)],
        physical_count=device.num_qubits,
    )
    circuit = topology_circuit()

    print(f"selected device: {device.display_name} [{device.name}]")
    print("backend status:", backend.status())
    print("selected physical path (logical q0, q1, q2):", physical_path)
    print("selected physical couplings:", physical_path[:2], physical_path[1:])
    print("Qiskit circuit:")
    print(circuit.draw(output="text"))

    # Compile and validate before creating any cloud task.
    artifact = compile_qiskit_circuit(
        circuit,
        device=device,
        options=CompilationOptions(initial_layout=layout, seed=SEED),
    )
    assert_compiled_mapping(artifact, backend, physical_path)
    print("compiled measurement mapping:", artifact.measurements)
    print("topology-validated QCIS:\n", artifact.qcis)
    print("PASS: Qiskit Target and cqlib physical topology/layout checks succeeded.")

    if not backend.is_available():
        print(
            f"SKIP: device {device.name!r} is currently {device.status.value}; "
            "the topology compilation test passed, but no task was submitted."
        )
        return

    started = perf_counter()
    job = backend.run(
        circuit,
        shots=shots,
        timeout=timeout,
        poll_interval=poll_interval,
        calibration="auto",
        initial_layout=layout,
        seed=SEED,
    )
    submitted_artifact = job.compilation_artifacts[0]
    assert_compiled_mapping(submitted_artifact, backend, physical_path)
    if submitted_artifact.qcis != artifact.qcis:
        raise AssertionError("submitted QCIS differs from the prevalidated QCIS")
    print("task IDs:", job.task_ids)
    print("job status after submission:", job.status())

    result = job.result(timeout=timeout, poll_interval=poll_interval)
    elapsed = perf_counter() - started
    counts = dict(result.get_counts())
    probabilities = dict(result.data()["probabilities"])
    print("final job status:", job.status())
    print("counts:", counts)
    print("probabilities:", probabilities)
    print(f"total submit + wait + result time: {elapsed:.3f} s")

    if sum(counts.values()) != shots:
        raise AssertionError(f"counts total does not equal shots: {counts}")
    if any(len(outcome.replace(" ", "")) != 3 for outcome in counts):
        raise AssertionError(f"cloud returned a non-three-bit outcome: {counts}")
    success = sum(count for outcome, count in counts.items() if outcome in IDEAL_OUTCOMES)
    success_probability = success / shots
    print("ideal outcomes:", sorted(IDEAL_OUTCOMES))
    print(f"ideal-outcome success probability: {success_probability:.3f}")
    if success_probability < MIN_SUCCESS_PROBABILITY:
        raise AssertionError(
            f"topology test success probability {success_probability:.3f} is below "
            f"{MIN_SUCCESS_PROBABILITY:.3f}"
        )
    print("PASS: real Tianyan topology/layout execution completed successfully.")


if __name__ == "__main__":
    main()
