"""Exact-state scaling comparison: Qiskit versus compiled cqlib QCIS.

Default matrix:
    qubits: 2, 4, 6, 8, 12, 14
    circuit depths: 5, 10, 15, 20

Both paths simulate the complete complex statevector without measurements or
shot sampling. The cqlib path remains Qiskit -> cqlib -> native QCIS -> cqlib
Statevector, so fidelity checks amplitudes and relative phases after native
compilation.

Smallest test:
    python examples/qiskit/06_scaling_simulator.py --smoke

Complete matrix:
    python examples/qiskit/06_scaling_simulator.py
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
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from cqlib_adapter.qiskit import CqlibSimulatorBackend

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
    unitary_depth: int
    cx_count: int
    dimension: int
    qiskit_seconds: float
    cqlib_seconds: float
    qiskit_norm: float
    cqlib_norm: float
    fidelity: float
    infidelity: float
    max_aligned_amplitude_error: float
    qcis_lines: int
    qcis_has_cz: bool
    qcis_has_cx: bool
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
    qubits: int,
    depth: int,
) -> QuantumCircuit:
    """Build a depth-exact parameterized brickwork-entangling circuit."""

    validate_case(qubits, depth)
    circuit = QuantumCircuit(
        qubits,
        name=f"scaling_q{qubits}_d{depth}",
    )
    blocks = depth // 5

    for block in range(blocks):
        for qubit in range(qubits):
            theta = pi * (0.21 + 0.04 * ((qubit + 2 * block) % 5))
            circuit.ry(theta, qubit)

        start = block % 2 if qubits > 2 else 0
        pairs = tuple((left, left + 1) for left in range(start, qubits - 1, 2))
        paired_qubits = {qubit for pair in pairs for qubit in pair}

        for control, target in pairs:
            circuit.cx(control, target)
        for qubit in range(qubits):
            if qubit not in paired_qubits:
                circuit.rz(
                    pi * (0.13 + 0.01 * qubit),
                    qubit,
                )

        for qubit in range(qubits):
            phi = pi * (0.09 + 0.03 * ((2 * qubit + block) % 7))
            circuit.rz(phi, qubit)

        for left, right in pairs:
            circuit.cx(right, left)
        for qubit in range(qubits):
            if qubit not in paired_qubits:
                circuit.rz(
                    -pi * (0.07 + 0.01 * qubit),
                    qubit,
                )

        for qubit in range(qubits):
            omega = pi * (0.17 + 0.025 * ((3 * qubit + block) % 6))
            circuit.ry(omega, qubit)

    actual_depth = circuit.depth()
    cx_count = int(circuit.count_ops().get("cx", 0))
    if actual_depth != depth:
        raise AssertionError(f"constructed depth {actual_depth}, expected {depth}")
    if cx_count == 0:
        raise AssertionError("benchmark circuit contains no two-qubit gate")
    return circuit


def run_qiskit_statevector(
    circuit: QuantumCircuit,
) -> tuple[np.ndarray, float]:
    """Simulate exact amplitudes with Qiskit QuantumInfo."""

    started = perf_counter()
    state = Statevector.from_instruction(circuit)
    elapsed = perf_counter() - started
    return np.asarray(state.data, dtype=np.complex128), elapsed


def run_cqlib_statevector(
    circuit: QuantumCircuit,
    seed: int,
) -> tuple[np.ndarray, float, str, tuple[int, ...]]:
    """Compile to native QCIS and simulate exact amplitudes with cqlib."""

    backend = CqlibSimulatorBackend(circuit.num_qubits)
    started = perf_counter()
    result = backend.run_statevector(circuit, seed=seed)
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
    """Maximum amplitude error after removing an irrelevant global phase."""

    overlap = np.vdot(reference, candidate)
    if abs(overlap) == 0:
        return float("inf")
    global_phase = overlap / abs(overlap)
    aligned = candidate / global_phase
    return float(np.max(np.abs(reference - aligned)))


def top_probabilities(
    state: np.ndarray,
    qubits: int,
    *,
    limit: int = 8,
) -> list[tuple[str, float]]:
    probabilities = np.abs(state) ** 2
    indices = np.argsort(probabilities)[::-1][:limit]
    return [(format(int(index), f"0{qubits}b"), float(probabilities[index])) for index in indices]


def execute_case(
    qubits: int,
    depth: int,
    seed: int,
    logger: logging.Logger,
    include_qcis: bool,
) -> CaseResult:
    circuit = build_benchmark_circuit(qubits, depth)
    qiskit_state, qiskit_seconds = run_qiskit_statevector(circuit)
    (
        cqlib_state,
        cqlib_seconds,
        qcis,
        physical_qubits,
    ) = run_cqlib_statevector(circuit, seed)

    if qiskit_state.shape != cqlib_state.shape:
        raise AssertionError(
            f"statevector dimension mismatch: {qiskit_state.shape} != {cqlib_state.shape}"
        )
    expected_physical_order = tuple(range(qubits))
    if physical_qubits != expected_physical_order:
        raise AssertionError(
            f"unexpected physical statevector order: {physical_qubits} != {expected_physical_order}"
        )

    qiskit_norm = float(np.vdot(qiskit_state, qiskit_state).real)
    cqlib_norm = float(np.vdot(cqlib_state, cqlib_state).real)
    raw_fidelity = float(
        state_fidelity(
            Statevector(qiskit_state),
            Statevector(cqlib_state),
            validate=True,
        )
    )
    fidelity = min(1.0, max(0.0, raw_fidelity))
    infidelity = max(0.0, 1.0 - fidelity)
    amplitude_error = aligned_amplitude_error(
        qiskit_state,
        cqlib_state,
    )

    qcis_operations = [
        line.split(maxsplit=1)[0].upper() for line in qcis.splitlines() if line.strip()
    ]
    qcis_has_cz = "CZ" in qcis_operations
    qcis_has_cx = "CX" in qcis_operations
    qcis_has_measurement = "M" in qcis_operations

    passed = (
        abs(qiskit_norm - 1.0) <= NORM_ATOL
        and abs(cqlib_norm - 1.0) <= NORM_ATOL
        and fidelity >= 1.0 - FIDELITY_ATOL
        and amplitude_error <= AMPLITUDE_ATOL
        and qcis_has_cz
        and not qcis_has_cx
        and not qcis_has_measurement
    )

    case = CaseResult(
        qubits=qubits,
        requested_depth=depth,
        unitary_depth=circuit.depth(),
        cx_count=int(circuit.count_ops().get("cx", 0)),
        dimension=len(qiskit_state),
        qiskit_seconds=qiskit_seconds,
        cqlib_seconds=cqlib_seconds,
        qiskit_norm=qiskit_norm,
        cqlib_norm=cqlib_norm,
        fidelity=fidelity,
        infidelity=infidelity,
        max_aligned_amplitude_error=amplitude_error,
        qcis_lines=len(qcis_operations),
        qcis_has_cz=qcis_has_cz,
        qcis_has_cx=qcis_has_cx,
        passed=passed,
    )

    logger.info(
        "CASE qubits=%d depth=%d dimension=%d",
        qubits,
        depth,
        case.dimension,
    )
    logger.info(
        "  source: unitary_depth=%d cx_count=%d",
        case.unitary_depth,
        case.cx_count,
    )
    logger.info(
        "  qiskit: seconds=%.6f norm=%.15f top_probabilities=%s",
        qiskit_seconds,
        qiskit_norm,
        top_probabilities(qiskit_state, qubits),
    )
    logger.info(
        "  cqlib:   seconds=%.6f norm=%.15f top_probabilities=%s",
        cqlib_seconds,
        cqlib_norm,
        top_probabilities(cqlib_state, qubits),
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
        "  QCIS: lines=%d has_CZ=%s has_CX=%s has_measurement=%s",
        case.qcis_lines,
        qcis_has_cz,
        qcis_has_cx,
        qcis_has_measurement,
    )
    if include_qcis:
        logger.info("  QCIS BEGIN\n%s\n  QCIS END", qcis)
    return case


def default_log_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "test-output" / f"qiskit_scaling_statevector_{timestamp}.log"


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cqlib_adapter.qiskit.statevector_scaling_example")
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
        help="write complete QCIS to the log",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only the smallest 2-qubit, depth-5 case",
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
    qubit_sizes = (2,) if args.smoke else tuple(args.qubits)
    depths = (5,) if args.smoke else tuple(args.depths)
    for qubits in qubit_sizes:
        for depth in depths:
            validate_case(qubits, depth)

    log_path = (args.log_file or default_log_path()).resolve()
    logger = configure_logger(log_path)
    logger.info("Qiskit/cqlib exact-state scaling comparison")
    logger.info("python=%s", sys.executable)
    logger.info(
        "versions: qiskit=%s cqlib=%s cqlib-adapter=%s",
        _package_version("qiskit"),
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
    if args.shots is not None:
        logger.warning(
            "--shots=%d is ignored: statevector simulation has no sampling",
            args.shots,
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
                    failures.append(f"q={qubits}, depth={depth}: statevector or QCIS check failed")
            except Exception as exc:
                logger.exception(
                    "CASE qubits=%d depth=%d raised an exception",
                    qubits,
                    depth,
                )
                failures.append(f"q={qubits}, depth={depth}: {type(exc).__name__}: {exc}")

    logger.info("SUMMARY")
    logger.info("  q  depth  dimension  qiskit(s)  cqlib(s)  fidelity           infidelity  result")
    for case in completed:
        logger.info(
            "  %2d  %5d  %9d  %9.6f  %8.6f  %.15f  %.3e  %s",
            case.qubits,
            case.requested_depth,
            case.dimension,
            case.qiskit_seconds,
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
