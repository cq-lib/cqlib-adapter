"""Explicitly authorized real Tianyan PennyLane QNode smoke test."""

from __future__ import annotations

import os
from time import perf_counter

import pennylane as qml

from cqlib_adapter.common import TianyanConnector
from cqlib_adapter.pennylane import TianyanDevice

DEFAULT_DEVICE = "tianyan176"

# Deliberately blank compatibility placeholder. Never put credentials in source.
TIANYAN_API_KEY = ""


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
    cloud_device = connector.resolve_device(DEFAULT_DEVICE)
    device = TianyanDevice(
        connector,
        cloud_device,
        wires=3,
        timeout=timeout,
        poll_interval=poll_interval,
        calibration="auto",
    )
    print(f"selected device: {cloud_device.display_name} [{cloud_device.name}]")
    print("status:", cloud_device.status.value)
    print("physical qubits:", cloud_device.num_qubits)
    print("native gates:", cloud_device.native_gates)
    print("topology edge count:", len(cloud_device.couplings))
    if not device.is_available():
        print("SKIP: selected Tianyan device is not currently running; no task submitted.")
        return

    @qml.qnode(device, shots=shots)
    def circuit():
        qml.PauliX(0)
        qml.Hadamard(1)
        # Wire 2 intentionally stays in |0>.
        return (
            qml.counts(wires=(0, 1, 2)),
            qml.probs(wires=(0, 1, 2)),
            qml.sample(wires=(0, 1, 2)),
        )

    print(qml.draw(circuit)())
    started = perf_counter()
    counts, probabilities, samples = circuit()
    elapsed = perf_counter() - started
    execution = device.last_executions[0]
    print("task IDs:", execution.task_ids)
    print("final status:", execution.status())
    print("submitted QCIS:\n", execution.qcis)
    print("counts:", counts)
    print("probabilities:", probabilities)
    print("first ten samples:\n", samples[:10])
    print(f"total compile + submit + wait + result time: {elapsed:.3f} s")

    if sum(counts.values()) != shots:
        raise AssertionError(f"counts total does not equal shots: {counts}")
    if samples.shape != (shots, 3):
        raise AssertionError(f"unexpected sample shape {samples.shape}")
    if abs(float(probabilities.sum()) - 1.0) > 1e-9:
        raise AssertionError("probabilities are not normalized")
    ideal_probability = (
        sum(count for outcome, count in counts.items() if outcome in {"100", "110"}) / shots
    )
    print(f"ideal-outcome probability (100 or 110): {ideal_probability:.3f}")
    if ideal_probability < 0.5:
        raise AssertionError("real-device result is inconsistent with X/H/I preparation")
    print("PASS: real Tianyan PennyLane QNode closed loop completed successfully.")


if __name__ == "__main__":
    main()
