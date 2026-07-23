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

"""Explicitly authorized real Tianyan CUDA-Q sampling smoke test."""

from __future__ import annotations

import os
from time import perf_counter

import cudaq

from cqlib_adapter.common import TianyanConnector
from cqlib_adapter.cudaq import TianyanExecutor

DEFAULT_DEVICE = "tianyan176"

# Deliberately blank compatibility placeholder. Never put credentials in source.
TIANYAN_API_KEY = ""


@cudaq.kernel
def cloud_smoke() -> None:
    qubits = cudaq.qvector(3)
    x(qubits[0])  # noqa: F821
    h(qubits[1])  # noqa: F821
    mz(qubits)  # noqa: F821


def positive_float(name: str, default: str) -> float:
    value = float(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> None:
    api_key = os.getenv("TIANYAN_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set TIANYAN_API_KEY in the current shell before running.")
    shots = int(positive_float("TIANYAN_TEST_SHOTS", "100"))
    timeout = positive_float("TIANYAN_TEST_TIMEOUT", "600")
    poll_interval = positive_float("TIANYAN_POLL_INTERVAL", "5")

    connector = TianyanConnector.login(api_key, save_credentials=False)
    device = connector.resolve_device(DEFAULT_DEVICE)
    executor = TianyanExecutor(
        connector,
        device,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    print(f"selected device: {device.display_name} [{device.name}]")
    print("status:", device.status.value)
    print("physical qubits:", device.num_qubits)
    print("native gates:", device.native_gates)
    print("topology edge count:", len(device.couplings))
    if not executor.is_available():
        print("SKIP: selected device is not running; no task was submitted.")
        return

    started = perf_counter()
    result = executor.sample(cloud_smoke, shots_count=shots)
    elapsed = perf_counter() - started
    job = executor.last_job
    if job is None:
        raise AssertionError("execution completed without a CUDA-Q job")
    counts = dict(result)
    print("task IDs:", job.task_ids)
    print("final status:", job.status())
    print("submitted QCIS:\n", job.qcis)
    print("CUDA-Q-compatible counts:", counts)
    print(f"compile + submit + wait + result time: {elapsed:.3f} s")

    if sum(counts.values()) != shots:
        raise AssertionError(f"counts total does not equal shots: {counts}")
    ideal_probability = (counts.get("100", 0) + counts.get("110", 0)) / shots
    if ideal_probability < 0.5:
        raise AssertionError("real-device result is inconsistent with X/H/I preparation")
    print("PASS: real Tianyan CUDA-Q closed loop completed successfully.")


if __name__ == "__main__":
    main()
