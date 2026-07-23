"""Validate a CUDA-Q kernel on one Tianyan physical three-qubit path."""

from __future__ import annotations

import os
import runpy
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import cudaq
from cqlib.device import Layout

from cqlib_adapter.common import CompilationOptions, NormalizedDevice, TianyanConnector
from cqlib_adapter.cudaq import TianyanExecutor, compile_cudaq_kernel

SEED = 43
IDEAL_OUTCOMES = {"100", "111"}
MIN_SUCCESS_PROBABILITY = 0.5


@cudaq.kernel
def topology_kernel() -> None:
    qubits = cudaq.qvector(3)
    x(qubits[0])  # noqa: F821
    h(qubits[1])  # noqa: F821
    x.ctrl(qubits[0], qubits[1])  # noqa: F821
    x.ctrl(qubits[1], qubits[2])  # noqa: F821
    mz(qubits)  # noqa: F821


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


def operation_qubits(operation: Any) -> tuple[int, ...]:
    return tuple(qubit.index for qubit in operation.qubits)


def assert_compiled_mapping(
    artifact: Any,
    device: NormalizedDevice,
    physical_path: tuple[int, int, int],
) -> None:
    selected = set(physical_path)
    path_edges = {
        frozenset(physical_path[:2]),
        frozenset(physical_path[1:]),
    }
    used: set[int] = set()
    two_qubit_edges: list[frozenset[int]] = []
    for operation in artifact.circuit.operations:
        qubits = operation_qubits(operation)
        used.update(qubits)
        if len(qubits) == 2:
            two_qubit_edges.append(frozenset(qubits))
    if not used <= selected:
        raise AssertionError(f"compiler escaped selected path: {sorted(used - selected)}")
    if not two_qubit_edges or not set(two_qubit_edges) <= path_edges:
        raise AssertionError(f"compiled gates violate selected topology: {two_qubit_edges}")
    if tuple(item.physical_qubit for item in artifact.measurements) != physical_path:
        raise AssertionError("CUDA-Q measurement mapping changed after physical layout")
    for left, right in pairwise(physical_path):
        if not device.supports_coupling(left, right, either_direction=True):
            raise AssertionError(f"selected device edge {(left, right)} is invalid")


def main() -> None:
    api_key, selector = load_cloud_config()
    shots = int(os.getenv("TIANYAN_TEST_SHOTS", "100"))
    timeout = float(os.getenv("TIANYAN_TEST_TIMEOUT", "600"))
    poll_interval = float(os.getenv("TIANYAN_POLL_INTERVAL", "5"))
    if shots <= 0 or timeout <= 0 or poll_interval <= 0:
        raise ValueError("shots, timeout and poll interval must be positive")

    connector = TianyanConnector.login(api_key, save_credentials=False)
    device = connector.resolve_device(selector)
    path = select_three_qubit_path(device)
    layout = Layout.from_pairs(
        [(logical, physical) for logical, physical in enumerate(path)],
        physical_count=device.num_qubits,
    )
    options = CompilationOptions(initial_layout=layout, seed=SEED)
    artifact = compile_cudaq_kernel(topology_kernel, device=device, options=options)
    assert_compiled_mapping(artifact, device, path)
    print(f"selected device: {device.display_name} [{device.name}]")
    print("selected physical path (CUDA-Q q0, q1, q2):", path)
    print("compiled measurement mapping:", artifact.measurements)
    print("topology-validated QCIS:\n", artifact.qcis)
    print("PASS: local CUDA-Q topology/layout validation succeeded.")
    if not device.available:
        print("SKIP: device is not running; validation passed but no task was submitted.")
        return

    executor = TianyanExecutor(
        connector,
        device,
        timeout=timeout,
        poll_interval=poll_interval,
        initial_layout=layout,
        seed=SEED,
    )
    started = perf_counter()
    result = executor.sample(topology_kernel, shots_count=shots)
    elapsed = perf_counter() - started
    job = executor.last_job
    if job is None:
        raise AssertionError("topology execution completed without a job")
    assert_compiled_mapping(job.artifact, device, path)
    if job.qcis != artifact.qcis:
        raise AssertionError("submitted QCIS differs from prevalidated QCIS")
    counts = dict(result)
    print("task IDs:", job.task_ids)
    print("CUDA-Q counts:", counts)
    print(f"total time: {elapsed:.3f} s")
    success = sum(count for outcome, count in counts.items() if outcome in IDEAL_OUTCOMES)
    if success / shots < MIN_SUCCESS_PROBABILITY:
        raise AssertionError("topology test success probability is below threshold")
    print("PASS: real Tianyan CUDA-Q topology execution completed successfully.")


if __name__ == "__main__":
    main()
