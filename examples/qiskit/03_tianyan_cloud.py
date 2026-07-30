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

"""Explicitly authorized real Tianyan Qiskit submission smoke test.

Credentials are read only from the ``TIANYAN_API_KEY`` environment variable.
They are never printed or stored. The device selector accepts either Tianyan's
internal backend name or its display name; by default this example selects
``tianyan176``.
"""

from __future__ import annotations

import os
from time import perf_counter

from qiskit import QuantumCircuit

from cqlib_adapter.common import TianyanConnector
from cqlib_adapter.qiskit import TianyanBackend

DEFAULT_DEVICE = "tianyan176"

# Deliberately blank compatibility placeholder. Never put credentials in source.
TIANYAN_API_KEY = ""


def positive_int(name: str, default: str) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def select_device(connector: TianyanConnector, query: str):
    """Resolve the requested device without downloading unrelated configs."""

    return connector.resolve_device(query)


def main() -> None:
    api_key = os.getenv("TIANYAN_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set TIANYAN_API_KEY in the current shell before running.")

    device_query = DEFAULT_DEVICE
    shots = positive_int("TIANYAN_TEST_SHOTS", "100")
    timeout = float(os.getenv("TIANYAN_TEST_TIMEOUT", "600"))
    poll_interval = float(os.getenv("TIANYAN_POLL_INTERVAL", "5"))
    if timeout <= 0 or poll_interval <= 0:
        raise ValueError("timeout and poll interval must be positive")

    connector = TianyanConnector.login(api_key, save_credentials=False)
    device = select_device(connector, device_query)
    backend = TianyanBackend.from_connector(connector, device.name)

    print(f"selected device: {device.display_name} [{device.name}]")
    print("backend status:", backend.status())
    print("qubits:", backend.num_qubits)
    print("native operations:", sorted(backend.target.operation_names))
    if not backend.is_available():
        raise RuntimeError(f"selected Tianyan device {device.display_name!r} is unavailable")

    circuit = QuantumCircuit(3, 3, name="qiskit-live-three-qubit-smoke")
    circuit.x(0)
    circuit.h(1)
    # Qubit 2 intentionally remains in |0>.
    circuit.measure([0, 1, 2], [0, 1, 2])
    print("Qiskit circuit:")
    print(circuit.draw(output="text"))

    started = perf_counter()
    job = backend.run(
        circuit,
        shots=shots,
        calibration="auto",
        timeout=timeout,
        poll_interval=poll_interval,
    )
    submitted = perf_counter()
    print("task IDs:", job.task_ids)
    print("job status after submission:", job.status())
    print("submitted QCIS:\n", job.qcis[0])

    result = job.result(timeout=timeout, poll_interval=poll_interval)
    finished = perf_counter()
    counts = dict(result.get_counts())
    probabilities = dict(result.data()["probabilities"])
    print("final job status:", job.status())
    print("counts:", counts)
    print("probabilities:", probabilities)
    print(f"submission time: {submitted - started:.3f} s")
    print(f"total submit + wait + result time: {finished - started:.3f} s")

    if not job.task_ids:
        raise AssertionError("Tianyan returned no task ID")
    invalid_outcomes = [
        outcome
        for outcome in counts
        if len(outcome.replace(" ", "")) != 3 or set(outcome.replace(" ", "")) - {"0", "1"}
    ]
    if invalid_outcomes:
        raise AssertionError(f"unexpected three-qubit outcomes: {invalid_outcomes}")
    if sum(counts.values()) != shots:
        raise AssertionError(f"counts total does not equal shots: {counts}")
    if abs(sum(probabilities.values()) - 1.0) > 1e-9:
        raise AssertionError(f"probabilities are not normalized: {probabilities}")
    print("PASS: real Tianyan Qiskit closed loop completed successfully.")


if __name__ == "__main__":
    main()
