"""Exact-state scaling comparison: Cirq versus compiled cqlib QCIS.

Default matrix:
    qubits: 2, 4, 6, 8, 12, 14
    circuit depths: 5, 10, 15, 20

No repetitions are used. Cirq's exact final state and the adapter's native-QCIS
cqlib Statevector are compared with pure-state fidelity.

Smallest test:
    python examples/cirq/07_scaling_statevector.py --smoke

Complete matrix:
    python examples/cirq/07_scaling_statevector.py
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

import cirq
import numpy as np
from cqlib.device import Layout

from cqlib_adapter.cirq import CqlibSimulatorSampler

DEFAULT_QUBITS = (2, 4, 6, 8, 12, 14)
DEFAULT_DEPTHS = (5, 10, 15, 20)
SEED = 2026
FIDELITY_ATOL = 1e-10
NORM_ATOL = 1e-10
AMPLITUDE_ATOL = 1e-9


@dataclass(frozen=True)
class CaseResult:
    qubits: int
    requested_depth: int
    operation_count: int
    cnot_count: int
    dimension: int
    cirq_seconds: float
    cqlib_seconds: float
    cirq_norm: float
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


def validate_case(qubits: int, depth: int) -> None:
    if qubits < 2:
        raise ValueError("at least two qubits are required for a two-qubit gate")
    if depth < 5 or depth % 5:
        raise ValueError("depth must be a positive multiple of 5")


def build_benchmark_circuit(
    qubit_count: int,
    depth: int,
) -> tuple[cirq.Circuit, tuple[cirq.LineQubit, ...]]:
    """Build a depth-exact Cirq moment structure with brickwork entangling."""

    validate_case(qubit_count, depth)
    qubits = tuple(cirq.LineQubit.range(qubit_count))
    moments: list[cirq.Moment] = []
    blocks = depth // 5

    for block in range(blocks):
        moments.append(
            cirq.Moment(
                cirq.ry(pi * (0.21 + 0.04 * ((index + 2 * block) % 5)))(qubit)
                for index, qubit in enumerate(qubits)
            )
        )

        start = block % 2 if qubit_count > 2 else 0
        pairs = tuple((left, left + 1) for left in range(start, qubit_count - 1, 2))
        paired = {index for pair in pairs for index in pair}
        forward: list[cirq.Operation] = [
            cirq.CNOT(qubits[control], qubits[target]) for control, target in pairs
        ]
        forward.extend(
            cirq.rz(pi * (0.13 + 0.01 * index))(qubits[index])
            for index in range(qubit_count)
            if index not in paired
        )
        moments.append(cirq.Moment(forward))

        moments.append(
            cirq.Moment(
                cirq.rz(pi * (0.09 + 0.03 * ((2 * index + block) % 7)))(qubit)
                for index, qubit in enumerate(qubits)
            )
        )

        reverse: list[cirq.Operation] = [
            cirq.CNOT(qubits[right], qubits[left]) for left, right in pairs
        ]
        reverse.extend(
            cirq.rz(-pi * (0.07 + 0.01 * index))(qubits[index])
            for index in range(qubit_count)
            if index not in paired
        )
        moments.append(cirq.Moment(reverse))

        moments.append(
            cirq.Moment(
                cirq.ry(pi * (0.17 + 0.025 * ((3 * index + block) % 6)))(qubit)
                for index, qubit in enumerate(qubits)
            )
        )

    circuit = cirq.Circuit(moments)
    if len(circuit) != depth:
        raise AssertionError(f"constructed depth {len(circuit)}, expected {depth}")
    cnot_count = sum(
        isinstance(operation.gate, cirq.CNotPowGate) for operation in circuit.all_operations()
    )
    if cnot_count == 0:
        raise AssertionError("benchmark circuit contains no two-qubit gate")
    return circuit, qubits


def run_cirq_statevector(
    circuit: cirq.Circuit,
    qubits: tuple[cirq.LineQubit, ...],
) -> tuple[np.ndarray, float]:
    """Execute exact amplitudes with Cirq's statevector simulator."""

    started = perf_counter()
    state = cirq.final_state_vector(
        circuit,
        qubit_order=qubits,
        dtype=np.complex128,
    )
    elapsed = perf_counter() - started
    return np.asarray(state, dtype=np.complex128), elapsed


def run_cqlib_statevector(
    circuit: cirq.Circuit,
    qubits: tuple[cirq.LineQubit, ...],
    seed: int,
) -> tuple[np.ndarray, float, str, tuple[int, ...]]:
    """Compile Cirq -> cqlib -> native QCIS and simulate amplitudes."""

    sampler = CqlibSimulatorSampler(
        len(qubits),
        initial_layout=Layout.from_pairs(
            [(logical, len(qubits) - 1 - logical) for logical in range(len(qubits))],
            physical_count=len(qubits),
        ),
        seed=seed,
        qubit_order=qubits,
    )
    started = perf_counter()
    result = sampler.run_statevector(
        circuit,
        qubit_order=qubits,
    )
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
    qubits: int,
    *,
    limit: int = 8,
) -> list[tuple[str, float]]:
    probabilities = np.abs(state) ** 2
    indices = np.argsort(probabilities)[::-1][:limit]
    return [
        (
            format(int(index), f"0{qubits}b"),
            float(probabilities[index]),
        )
        for index in indices
    ]


def execute_case(
    qubit_count: int,
    depth: int,
    seed: int,
    logger: logging.Logger,
    include_qcis: bool,
) -> CaseResult:
    circuit, qubits = build_benchmark_circuit(
        qubit_count,
        depth,
    )
    reference, cirq_seconds = run_cirq_statevector(
        circuit,
        qubits,
    )
    (
        candidate,
        cqlib_seconds,
        qcis,
        physical_qubits,
    ) = run_cqlib_statevector(
        circuit,
        qubits,
        seed,
    )

    if reference.shape != candidate.shape:
        raise AssertionError(
            f"statevector dimension mismatch: {reference.shape} != {candidate.shape}"
        )
    expected_layout = tuple(reversed(range(qubit_count)))
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
    operations = tuple(circuit.all_operations())
    cnot_count = sum(isinstance(operation.gate, cirq.CNotPowGate) for operation in operations)

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
        qubits=qubit_count,
        requested_depth=depth,
        operation_count=len(operations),
        cnot_count=cnot_count,
        dimension=len(reference),
        cirq_seconds=cirq_seconds,
        cqlib_seconds=cqlib_seconds,
        cirq_norm=reference_norm,
        cqlib_norm=candidate_norm,
        fidelity=fidelity,
        infidelity=infidelity,
        max_aligned_amplitude_error=amplitude_error,
        qcis_lines=len(qcis_operations),
        passed=passed,
    )

    logger.info(
        "CASE qubits=%d depth=%d dimension=%d",
        qubit_count,
        depth,
        case.dimension,
    )
    logger.info(
        "  source: operation_count=%d cnot_count=%d",
        case.operation_count,
        case.cnot_count,
    )
    logger.info(
        "  Cirq:  seconds=%.6f norm=%.15f top_probabilities=%s",
        cirq_seconds,
        reference_norm,
        top_probabilities(reference, qubit_count),
    )
    logger.info(
        "  cqlib: seconds=%.6f norm=%.15f top_probabilities=%s",
        cqlib_seconds,
        candidate_norm,
        top_probabilities(candidate, qubit_count),
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
    return project_root / "test-output" / f"cirq_scaling_statevector_{timestamp}.log"


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cqlib_adapter.cirq.statevector_scaling")
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
        "--qubits",
        type=parse_integer_list,
        default=DEFAULT_QUBITS,
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
        help="run only the 2-qubit, depth-5 case",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    qubit_sizes = (2,) if args.smoke else tuple(args.qubits)
    depths = (5,) if args.smoke else tuple(args.depths)
    for qubits in qubit_sizes:
        for depth in depths:
            validate_case(qubits, depth)

    log_path = (args.log_file or default_log_path()).resolve()
    logger = configure_logger(log_path)
    logger.info("Cirq/cqlib exact-state scaling comparison")
    logger.info("python=%s", sys.executable)
    logger.info(
        "versions: Cirq=%s cqlib=%s cqlib-adapter=%s",
        _package_version("cirq-core"),
        _package_version("cqlib"),
        _package_version("cqlib-adapter"),
    )
    logger.info(
        "matrix: qubits=%s depths=%s seed=%d cases=%d",
        qubit_sizes,
        depths,
        args.seed,
        len(qubit_sizes) * len(depths),
    )
    logger.info(
        "thresholds: fidelity>=%.12f norm_atol=%.1e amplitude_atol=%.1e",
        1.0 - FIDELITY_ATOL,
        NORM_ATOL,
        AMPLITUDE_ATOL,
    )
    ignored_sampling = args.repetitions if args.repetitions is not None else args.shots
    if ignored_sampling is not None:
        logger.warning(
            "sampling count %d is ignored: statevector simulation has no sampling",
            ignored_sampling,
        )
    logger.info("log_file=%s", log_path)

    completed: list[CaseResult] = []
    failures: list[str] = []
    for qubits in qubit_sizes:
        for depth in depths:
            try:
                case = execute_case(
                    qubits,
                    depth,
                    args.seed,
                    logger,
                    args.include_qcis,
                )
                completed.append(case)
                if not case.passed:
                    failures.append(
                        f"qubits={qubits}, depth={depth}: fidelity or QCIS check failed"
                    )
            except Exception as exc:
                logger.exception(
                    "CASE qubits=%d depth=%d raised an exception",
                    qubits,
                    depth,
                )
                failures.append(f"qubits={qubits}, depth={depth}: {type(exc).__name__}: {exc}")

    logger.info("SUMMARY")
    logger.info("  qubits depth dimension Cirq(s) cqlib(s) fidelity           infidelity result")
    for case in completed:
        logger.info(
            "  %6d %5d %9d %7.6f %8.6f %.15f %.3e %s",
            case.qubits,
            case.requested_depth,
            case.dimension,
            case.cirq_seconds,
            case.cqlib_seconds,
            case.fidelity,
            case.infidelity,
            "PASS" if case.passed else "FAIL",
        )

    total_cases = len(qubit_sizes) * len(depths)
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
