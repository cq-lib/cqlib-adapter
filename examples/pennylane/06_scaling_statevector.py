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

"""Exact-state scaling comparison: PennyLane versus compiled cqlib QCIS.

Default matrix:
    wires: 2, 4, 6, 8, 12, 14
    circuit depths: 5, 10, 15, 20

No shots are used. PennyLane default.qubit and the adapter's native-QCIS cqlib
Statevector are compared with pure-state fidelity.

Smallest test:
    python examples/pennylane/06_scaling_statevector.py --smoke

Complete matrix:
    python examples/pennylane/06_scaling_statevector.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from math import pi
from pathlib import Path
from time import perf_counter

import numpy as np
import pennylane as qml
from cqlib.device import Layout
from pennylane.tape import QuantumScript

from cqlib_adapter.pennylane import CqlibSimulatorDevice

DEFAULT_WIRES = (2, 4, 6, 8, 12, 14)
DEFAULT_DEPTHS = (5, 10, 15, 20)
SEED = 2026
FIDELITY_ATOL = 1e-10
NORM_ATOL = 1e-10
AMPLITUDE_ATOL = 1e-9


@dataclass(frozen=True)
class CaseResult:
    wires: int
    requested_depth: int
    operation_count: int
    cnot_count: int
    dimension: int
    pennylane_seconds: float
    cqlib_seconds: float
    pennylane_norm: float
    cqlib_norm: float
    fidelity: float
    infidelity: float
    max_aligned_amplitude_error: float
    qcis_lines: int
    passed: bool


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def parse_integer_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def validate_case(wires: int, depth: int) -> None:
    if wires < 2:
        raise ValueError("at least two wires are required for a two-qubit gate")
    if depth < 5 or depth % 5:
        raise ValueError("depth must be a positive multiple of 5")


def build_benchmark_tape(
    wires: int,
    depth: int,
) -> QuantumScript:
    """Build the same five-layer brickwork ansatz as the Qiskit example."""

    validate_case(wires, depth)
    operations: list[qml.operation.Operator] = []
    blocks = depth // 5

    for block in range(blocks):
        for wire in range(wires):
            theta = pi * (0.21 + 0.04 * ((wire + 2 * block) % 5))
            operations.append(qml.RY(theta, wires=wire))

        start = block % 2 if wires > 2 else 0
        pairs = tuple((left, left + 1) for left in range(start, wires - 1, 2))
        paired_wires = {wire for pair in pairs for wire in pair}

        for control, target in pairs:
            operations.append(qml.CNOT(wires=(control, target)))
        for wire in range(wires):
            if wire not in paired_wires:
                operations.append(
                    qml.RZ(
                        pi * (0.13 + 0.01 * wire),
                        wires=wire,
                    )
                )

        for wire in range(wires):
            phi = pi * (0.09 + 0.03 * ((2 * wire + block) % 7))
            operations.append(qml.RZ(phi, wires=wire))

        for left, right in pairs:
            operations.append(qml.CNOT(wires=(right, left)))
        for wire in range(wires):
            if wire not in paired_wires:
                operations.append(
                    qml.RZ(
                        -pi * (0.07 + 0.01 * wire),
                        wires=wire,
                    )
                )

        for wire in range(wires):
            omega = pi * (0.17 + 0.025 * ((3 * wire + block) % 6))
            operations.append(qml.RY(omega, wires=wire))

    cnot_count = sum(operation.name == "CNOT" for operation in operations)
    if cnot_count == 0:
        raise AssertionError("benchmark tape contains no two-qubit gate")
    return QuantumScript(tuple(operations), ())


def run_pennylane_statevector(
    tape: QuantumScript,
    wires: int,
) -> tuple[np.ndarray, float]:
    """Execute exact amplitudes with PennyLane default.qubit."""

    reference_tape = QuantumScript(
        tuple(tape.operations),
        (qml.state(),),
    )
    device = qml.device(
        "default.qubit",
        wires=wires,
        shots=None,
    )
    started = perf_counter()
    state = qml.execute(
        (reference_tape,),
        device,
        diff_method=None,
    )[0]
    elapsed = perf_counter() - started
    return np.asarray(state, dtype=np.complex128), elapsed


def run_cqlib_statevector(
    tape: QuantumScript,
    wires: int,
    seed: int,
) -> tuple[np.ndarray, float, str, tuple[int, ...]]:
    """Compile PennyLane -> cqlib -> native QCIS and simulate amplitudes."""

    device = CqlibSimulatorDevice(
        wires=wires,
        initial_layout=Layout.from_pairs(
            [(logical, wires - 1 - logical) for logical in range(wires)],
            physical_count=wires,
        ),
        seed=seed,
    )
    started = perf_counter()
    result = device.run_statevector(tape)
    elapsed = perf_counter() - started
    return (
        np.asarray(result.data, dtype=np.complex128),
        elapsed,
        result.qcis,
        result.physical_qubits,
    )


def aligned_amplitude_error(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    overlap = np.vdot(reference, candidate)
    if abs(overlap) == 0:
        return float("inf")
    aligned = candidate / (overlap / abs(overlap))
    return float(np.max(np.abs(reference - aligned)))


def top_probabilities(
    state: np.ndarray,
    wires: int,
    *,
    limit: int = 8,
) -> list[tuple[str, float]]:
    probabilities = np.abs(state) ** 2
    indices = np.argsort(probabilities)[::-1][:limit]
    return [
        (
            format(int(index), f"0{wires}b"),
            float(probabilities[index]),
        )
        for index in indices
    ]


def execute_case(
    wires: int,
    depth: int,
    seed: int,
    logger: logging.Logger,
    include_qcis: bool,
) -> CaseResult:
    tape = build_benchmark_tape(wires, depth)
    reference, pennylane_seconds = run_pennylane_statevector(
        tape,
        wires,
    )
    (
        candidate,
        cqlib_seconds,
        qcis,
        physical_qubits,
    ) = run_cqlib_statevector(tape, wires, seed)

    if reference.shape != candidate.shape:
        raise AssertionError(
            f"statevector dimension mismatch: {reference.shape} != {candidate.shape}"
        )
    expected_layout = tuple(reversed(range(wires)))
    if physical_qubits != expected_layout:
        raise AssertionError(
            f"compiled layout {physical_qubits} did not preserve expected {expected_layout}"
        )

    reference_norm = float(np.vdot(reference, reference).real)
    candidate_norm = float(np.vdot(candidate, candidate).real)
    raw_fidelity = float(
        abs(np.vdot(reference, candidate)) ** 2 / (reference_norm * candidate_norm)
    )
    fidelity = min(1.0, max(0.0, raw_fidelity))
    infidelity = max(0.0, 1.0 - fidelity)
    amplitude_error = aligned_amplitude_error(
        reference,
        candidate,
    )

    qcis_operations = [
        line.split(maxsplit=1)[0].upper() for line in qcis.splitlines() if line.strip()
    ]
    has_cz = "CZ" in qcis_operations
    has_cx = "CX" in qcis_operations
    has_measurement = "M" in qcis_operations
    cnot_count = sum(operation.name == "CNOT" for operation in tape.operations)

    passed = (
        abs(reference_norm - 1.0) <= NORM_ATOL
        and abs(candidate_norm - 1.0) <= NORM_ATOL
        and fidelity >= 1.0 - FIDELITY_ATOL
        and amplitude_error <= AMPLITUDE_ATOL
        and has_cz
        and not has_cx
        and not has_measurement
    )
    case = CaseResult(
        wires=wires,
        requested_depth=depth,
        operation_count=len(tape.operations),
        cnot_count=cnot_count,
        dimension=len(reference),
        pennylane_seconds=pennylane_seconds,
        cqlib_seconds=cqlib_seconds,
        pennylane_norm=reference_norm,
        cqlib_norm=candidate_norm,
        fidelity=fidelity,
        infidelity=infidelity,
        max_aligned_amplitude_error=amplitude_error,
        qcis_lines=len(qcis_operations),
        passed=passed,
    )

    logger.info(
        "CASE wires=%d depth=%d dimension=%d",
        wires,
        depth,
        case.dimension,
    )
    logger.info(
        "  source: operation_count=%d cnot_count=%d",
        case.operation_count,
        case.cnot_count,
    )
    logger.info(
        "  PennyLane: seconds=%.6f norm=%.15f top_probabilities=%s",
        pennylane_seconds,
        reference_norm,
        top_probabilities(reference, wires),
    )
    logger.info(
        "  cqlib:     seconds=%.6f norm=%.15f top_probabilities=%s",
        cqlib_seconds,
        candidate_norm,
        top_probabilities(candidate, wires),
    )
    logger.info(
        "  compare: fidelity=%.15f infidelity=%.3e "
        "max_phase_aligned_amplitude_error=%.3e status=%s",
        fidelity,
        infidelity,
        amplitude_error,
        "PASS" if passed else "FAIL",
    )
    logger.info(
        "  QCIS: lines=%d logical_to_physical=%s has_CZ=%s has_CX=%s has_measurement=%s",
        case.qcis_lines,
        physical_qubits,
        has_cz,
        has_cx,
        has_measurement,
    )
    if include_qcis:
        logger.info("  QCIS BEGIN\n%s\n  QCIS END", qcis)
    return case


def default_log_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "test-output" / f"pennylane_scaling_statevector_{timestamp}.log"


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cqlib_adapter.pennylane.statevector_scaling")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wires",
        type=parse_integer_list,
        default=DEFAULT_WIRES,
    )
    parser.add_argument(
        "--depths",
        type=parse_integer_list,
        default=DEFAULT_DEPTHS,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--include-qcis",
        action="store_true",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only the 2-wire, depth-5 case",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    wire_sizes = (2,) if args.smoke else tuple(args.wires)
    depths = (5,) if args.smoke else tuple(args.depths)
    for wires in wire_sizes:
        for depth in depths:
            validate_case(wires, depth)

    log_path = (args.log_file or default_log_path()).resolve()
    logger = configure_logger(log_path)
    logger.info("PennyLane/cqlib exact-state scaling comparison")
    logger.info("python=%s", sys.executable)
    logger.info(
        "versions: PennyLane=%s cqlib=%s cqlib-adapter=%s",
        _package_version("pennylane"),
        _package_version("cqlib"),
        _package_version("cqlib-adapter"),
    )
    logger.info(
        "matrix: wires=%s depths=%s seed=%d cases=%d",
        wire_sizes,
        depths,
        args.seed,
        len(wire_sizes) * len(depths),
    )
    logger.info(
        "thresholds: fidelity>=%.12f norm_atol=%.1e amplitude_atol=%.1e",
        1.0 - FIDELITY_ATOL,
        NORM_ATOL,
        AMPLITUDE_ATOL,
    )
    if args.shots is not None:
        logger.warning(
            "--shots=%d is ignored: statevector simulation has no sampling",
            args.shots,
        )
    logger.info("log_file=%s", log_path)

    completed: list[CaseResult] = []
    failures: list[str] = []
    for wires in wire_sizes:
        for depth in depths:
            try:
                case = execute_case(
                    wires,
                    depth,
                    args.seed,
                    logger,
                    args.include_qcis,
                )
                completed.append(case)
                if not case.passed:
                    failures.append(
                        f"wires={wires}, depth={depth}: statevector or QCIS check failed"
                    )
            except Exception as exc:
                logger.exception(
                    "CASE wires=%d depth=%d raised an exception",
                    wires,
                    depth,
                )
                failures.append(f"wires={wires}, depth={depth}: {type(exc).__name__}: {exc}")

    logger.info("SUMMARY")
    logger.info(
        "  wires depth dimension PennyLane(s) cqlib(s) fidelity           infidelity result"
    )
    for case in completed:
        logger.info(
            "  %5d %5d %9d %12.6f %8.6f %.15f %.3e %s",
            case.wires,
            case.requested_depth,
            case.dimension,
            case.pennylane_seconds,
            case.cqlib_seconds,
            case.fidelity,
            case.infidelity,
            "PASS" if case.passed else "FAIL",
        )

    total_cases = len(wire_sizes) * len(depths)
    if failures:
        logger.error(
            "FAILED %d/%d cases",
            len(failures),
            total_cases,
        )
        for failure in failures:
            logger.error("  %s", failure)
        logger.error("Inspect the full log at %s", log_path)
        return 1

    logger.info(
        "PASS: all %d cases passed exact-state fidelity validation",
        len(completed),
    )
    logger.info("Full log: %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
