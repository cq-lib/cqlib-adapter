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

"""Real Tianyan topology/layout smoke test for the PennyLane adapter.

The script reads the API key from the current shell and reuses the device
selector configured in 03_tianyan_cloud.py.
It chooses a connected three-qubit physical path from the cqlib device
snapshot, maps PennyLane wires 0/1/2 onto that path, validates every compiled
two-qubit gate and measurement locally, and only then submits a real task.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from time import perf_counter
from typing import Any

import pennylane as qml
from cqlib.device import Layout
from pennylane.tape import QuantumScript

from cqlib_adapter.common import CompilationOptions, NormalizedDevice, TianyanConnector
from cqlib_adapter.pennylane import TianyanDevice, compile_pennylane_circuit

SEED = 41
IDEAL_OUTCOMES = {"100", "111"}
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
    """Select a deterministic usable left-center-right physical path."""

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


def topology_tape(shots: int) -> QuantumScript:
    """Prepare wire0=1 and a Bell-correlated wire1/wire2 pair."""

    return QuantumScript(
        [
            qml.PauliX(0),
            qml.Hadamard(1),
            qml.CNOT((0, 1)),
            qml.CNOT((1, 2)),
        ],
        [
            qml.counts(wires=(0, 1, 2)),
            qml.probs(wires=(0, 1, 2)),
            qml.sample(wires=(0, 1, 2)),
        ],
        shots=shots,
    )


def operation_qubits(operation: Any) -> tuple[int, ...]:
    return tuple(qubit.index for qubit in operation.qubits)


def assert_compiled_mapping(
    artifact: Any,
    device: NormalizedDevice,
    physical_path: tuple[int, int, int],
) -> None:
    """Prove compiled gates and measurements obey the selected cqlib topology."""

    left, center, right = physical_path
    selected = set(physical_path)
    expected_edges = {frozenset((left, center)), frozenset((center, right))}
    device_edges = {
        frozenset((edge.source, edge.target))
        for edge in device.couplings
        if edge.source in device.usable_qubits and edge.target in device.usable_qubits
    }
    if not expected_edges <= device_edges:
        raise AssertionError(
            f"selected path is not present in the cqlib device topology: expected={expected_edges}"
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
            "compiler escaped the requested physical path: "
            f"used={sorted(compiled_qubits)}, selected={sorted(selected)}"
        )
    if not compiled_two_qubit_edges:
        raise AssertionError("compiled circuit contains no two-qubit operation")
    if not set(compiled_two_qubit_edges) <= expected_edges:
        raise AssertionError(
            "compiled two-qubit operation violates the selected topology path: "
            f"actual={compiled_two_qubit_edges}, expected={expected_edges}"
        )

    measured = tuple(item.physical_qubit for item in artifact.measurements)
    if measured != physical_path:
        raise AssertionError(
            "PennyLane logical-to-physical measurement mapping changed: "
            f"expected={physical_path}, actual={measured}"
        )
    classical = tuple(item.classical_bit for item in artifact.measurements)
    if classical != (0, 1, 2):
        raise AssertionError(f"unexpected PennyLane classical-bit mapping: {classical}")


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
    tape = topology_tape(shots)

    print(f"selected device: {cloud_device.display_name} [{cloud_device.name}]")
    print("device status:", cloud_device.status.value)
    print("selected physical path (PennyLane wires 0, 1, 2):", physical_path)
    print("selected physical couplings:", physical_path[:2], physical_path[1:])
    print("logical circuit:")
    print("  X(wire 0); H(wire 1); CNOT(0,1); CNOT(1,2); measure wires 0,1,2")

    options = CompilationOptions(initial_layout=layout, seed=SEED)
    artifact = compile_pennylane_circuit(
        tape,
        wire_order=(0, 1, 2),
        device=cloud_device,
        options=options,
    )
    assert_compiled_mapping(artifact, cloud_device, physical_path)
    print("compiled measurement mapping:", artifact.measurements)
    print("topology-validated QCIS:\n", artifact.qcis)
    print("PASS: PennyLane wires were mapped onto the selected cqlib physical path.")

    if not cloud_device.available:
        print(
            f"SKIP: device {cloud_device.name!r} is currently "
            f"{cloud_device.status.value}; local topology validation passed, "
            "but no task was submitted."
        )
        return

    device = TianyanDevice(
        connector,
        cloud_device,
        wires=3,
        timeout=timeout,
        poll_interval=poll_interval,
        calibration="auto",
        initial_layout=layout,
        seed=SEED,
    )

    @qml.qnode(device, shots=shots)
    def circuit():
        qml.PauliX(0)
        qml.Hadamard(1)
        qml.CNOT((0, 1))
        qml.CNOT((1, 2))
        return (
            qml.counts(wires=(0, 1, 2)),
            qml.probs(wires=(0, 1, 2)),
            qml.sample(wires=(0, 1, 2)),
        )

    started = perf_counter()
    counts, probabilities, samples = circuit()
    elapsed = perf_counter() - started
    execution = device.last_executions[0]
    submitted_artifact = execution.artifact
    assert_compiled_mapping(submitted_artifact, cloud_device, physical_path)
    if submitted_artifact.qcis != artifact.qcis:
        raise AssertionError("submitted QCIS differs from the prevalidated QCIS")

    print("task IDs:", execution.task_ids)
    print("final status:", execution.status())
    print("counts in PennyLane wire order:", counts)
    print("probabilities:", probabilities)
    print("first ten samples:\n", samples[:10])
    print(f"total compile + submit + wait + result time: {elapsed:.3f} s")

    if sum(counts.values()) != shots:
        raise AssertionError(f"counts total does not equal shots: {counts}")
    if any(len(outcome) != 3 for outcome in counts):
        raise AssertionError(f"cloud returned a non-three-bit PennyLane outcome: {counts}")
    if samples.shape != (shots, 3):
        raise AssertionError(f"unexpected sample shape {samples.shape}")
    if abs(float(probabilities.sum()) - 1.0) > 1e-9:
        raise AssertionError("probabilities are not normalized")
    success = sum(count for outcome, count in counts.items() if outcome in IDEAL_OUTCOMES)
    success_probability = success / shots
    print("ideal PennyLane outcomes:", sorted(IDEAL_OUTCOMES))
    print(f"ideal-outcome success probability: {success_probability:.3f}")
    if success_probability < MIN_SUCCESS_PROBABILITY:
        raise AssertionError(
            f"topology test success probability {success_probability:.3f} is below "
            f"{MIN_SUCCESS_PROBABILITY:.3f}"
        )
    print("PASS: real Tianyan PennyLane topology/layout execution completed successfully.")


if __name__ == "__main__":
    main()
