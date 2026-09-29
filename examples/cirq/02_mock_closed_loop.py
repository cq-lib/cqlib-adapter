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

"""No-network Cirq Sampler/result contract test with preset counts.

This mock does not simulate the submitted circuit; it returns the preset
counts verbatim. It is the only offline execution path, intended for
transport/job/result contract validation.
"""

from __future__ import annotations

import cirq

from cqlib_adapter.cirq.testing import ResultSpec, make_cirq_sampler


def main() -> None:
    shots = 20
    sampler, cloud = make_cirq_sampler(
        [ResultSpec({"110": shots}, (0, 1, 2), status_ready=True)],
        size=3,
    )
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(q1),
        cirq.X(q2),
        cirq.measure(q0, key="left"),
        cirq.measure(q1, q2, key="pair"),
    )

    result = sampler.run(circuit, repetitions=shots)
    print("Cirq measurements:", result.measurements)
    print("left histogram:", result.histogram(key="left"))
    print("pair histogram:", result.histogram(key="pair"))
    print("task IDs:", sampler.last_task_ids)
    print("submitted QCIS:\n", sampler.last_qcis[0])
    print("mock transport call:", cloud.calls[0])

    if result.histogram(key="left") != {0: shots}:
        raise AssertionError("left measurement key is incorrect")
    if result.histogram(key="pair") != {3: shots}:
        raise AssertionError("pair measurement key or bit order is incorrect")
    print("PASS: mock Sampler/ResultDict contract succeeded; semantics not evaluated.")


if __name__ == "__main__":
    main()
