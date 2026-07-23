"""No-network CUDA-Q job/sample-result contract test with preset counts.

This mock does not simulate the submitted kernel. Use the local cqlib examples
for quantum-semantic validation.
"""

from __future__ import annotations

import cudaq

from cqlib_adapter.common import JobState
from cqlib_adapter.cudaq.testing import ResultSpec, make_cudaq_executor


@cudaq.kernel
def state_110() -> None:
    qubits = cudaq.qvector(3)
    x(qubits[0])  # noqa: F821
    x(qubits[1])  # noqa: F821
    mz(qubits)  # noqa: F821


def main() -> None:
    shots = 20
    # Tianyan/cqlib physical MSB-left payload 011 maps to CUDA-Q q0-left 110.
    executor, cloud = make_cudaq_executor(
        [ResultSpec({"011": shots}, (0, 1, 2))],
        size=3,
    )
    job = executor.sample_async(state_110, shots_count=shots)
    print("task ID:", job.task_id)
    print("initial status:", job.status())
    if job.status().state is not JobState.SUBMITTED:
        raise AssertionError("mock job should initially be submitted")

    result = job.get(timeout=2, poll_interval=0.01)
    print("CUDA-Q-compatible counts:", dict(result))
    print("probability(110):", result.probability("110"))
    print("submitted QCIS:\n", job.qcis)
    print("mock transport call:", cloud.calls[0])
    if dict(result) != {"110": shots}:
        raise AssertionError("mock closed loop changed CUDA-Q bit order")
    print("PASS: mock async-job/sample contract succeeded; semantics not evaluated.")


if __name__ == "__main__":
    main()
