"""Explicitly authorized real Tianyan Cirq Sampler smoke test."""

from __future__ import annotations

import os
from time import perf_counter

import cirq

from cqlib_adapter.cirq import TianyanSampler
from cqlib_adapter.common import TianyanConnector

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
    sampler = TianyanSampler(
        connector,
        cloud_device,
        timeout=timeout,
        poll_interval=poll_interval,
        calibration="auto",
    )
    print(f"selected device: {cloud_device.display_name} [{cloud_device.name}]")
    print("status:", cloud_device.status.value)
    print("physical qubits:", cloud_device.num_qubits)
    print("native gates:", cloud_device.native_gates)
    print("topology edge count:", len(cloud_device.couplings))
    if not sampler.is_available():
        print("SKIP: selected Tianyan device is not currently running; no task submitted.")
        return

    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.H(q1),
        # q2 intentionally stays in |0>.
        cirq.measure(q0, q1, q2, key="state"),
    )
    print("Cirq circuit:")
    print(circuit)

    started = perf_counter()
    result = sampler.run(circuit, repetitions=shots)
    elapsed = perf_counter() - started
    execution = sampler.last_executions[0]
    histogram = dict(result.histogram(key="state"))
    print("task IDs:", execution.task_ids)
    print("final status:", execution.status())
    print("submitted QCIS:\n", execution.qcis)
    print("measurement array:\n", result.measurements["state"][:10])
    print("histogram:", histogram)
    print(f"total compile + submit + wait + result time: {elapsed:.3f} s")

    if sum(histogram.values()) != shots:
        raise AssertionError(f"histogram total does not equal repetitions: {histogram}")
    ideal = histogram.get(4, 0) + histogram.get(6, 0)
    probability = ideal / shots
    print(f"ideal-outcome probability (100 or 110): {probability:.3f}")
    if probability < 0.5:
        raise AssertionError("real-device result is inconsistent with X/H/I preparation")
    print("PASS: real Tianyan Cirq Sampler closed loop completed successfully.")


if __name__ == "__main__":
    main()
