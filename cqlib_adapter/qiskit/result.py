"""Convert framework-neutral Tianyan data to Qiskit Result objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qiskit.result import Result

from cqlib_adapter.common import (
    AdapterResultError,
    CanonicalResult,
    CompilationArtifact,
)


def _hex_outcome(bitstring: str) -> str:
    return hex(int(bitstring, 2)) if bitstring else "0x0"


def _experiment(
    canonical: CanonicalResult,
    artifact: CompilationArtifact,
) -> dict[str, Any]:
    if canonical.num_classical_bits != artifact.num_classical_bits:
        raise AdapterResultError("canonical result width does not match compilation metadata")
    header: dict[str, Any] = {
        "name": artifact.metadata.circuit_name or canonical.task_id,
        "memory_slots": canonical.num_classical_bits,
        "metadata": dict(artifact.metadata.extras),
        "task_id": canonical.task_id,
    }
    if artifact.metadata.measurements.register_sizes:
        header["creg_sizes"] = [
            [name, size] for name, size in artifact.metadata.measurements.register_sizes.items()
        ]

    counts = {_hex_outcome(outcome): count for outcome, count in canonical.counts.items()}
    memory: list[str] = []
    for outcome, count in sorted(canonical.counts.items()):
        memory.extend([_hex_outcome(outcome)] * count)

    return {
        "shots": canonical.shots,
        "success": True,
        "status": "DONE",
        "header": header,
        "data": {
            "counts": counts,
            "memory": memory,
            "probabilities": dict(canonical.probabilities),
        },
    }


def canonical_to_qiskit_result(
    canonical_results: Sequence[CanonicalResult],
    artifacts: Sequence[CompilationArtifact],
    *,
    backend_name: str,
    backend_version: str = "2.0.0",
) -> Result:
    """Build a standard Qiskit Result for one Tianyan batch."""

    if len(canonical_results) != len(artifacts):
        raise AdapterResultError("result count does not match the submitted Qiskit circuit count")
    if not canonical_results:
        raise AdapterResultError("at least one canonical result is required")
    job_id = ",".join(result.task_id for result in canonical_results)
    return Result.from_dict(
        {
            "backend_name": backend_name,
            "backend_version": backend_version,
            "qobj_id": None,
            "job_id": job_id,
            "success": True,
            "status": "COMPLETED",
            "results": [
                _experiment(result, artifact)
                for result, artifact in zip(
                    canonical_results,
                    artifacts,
                    strict=True,
                )
            ],
        }
    )


__all__ = ["canonical_to_qiskit_result"]
