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

"""No-network Qiskit transport/job/result contract test with preset counts.

This mock does not simulate the submitted circuit. Use the local cqlib examples
for quantum-semantic validation.
"""

from qiskit import QuantumCircuit

from cqlib_adapter.qiskit.testing import ResultSpec, make_qiskit_backend

backend, mock_cloud = make_qiskit_backend(
    [ResultSpec({"00": 70, "11": 30}, (0, 1))],
    size=2,
)
circuit = QuantumCircuit(2, 2, name="bell-mock")
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

job = backend.run(circuit, shots=100, poll_interval=0.01)
print("task IDs:", job.task_ids)
print("job status before wait:", job.status())
print("submitted QCIS:\n", job.qcis[0])

result = job.result(timeout=2, poll_interval=0.01)
print("Qiskit counts:", result.get_counts())
print("Qiskit memory rows:", len(result.get_memory()))
print("probabilities:", result.data()["probabilities"])
print("mock transport calls:", mock_cloud.calls)
if result.get_counts() != {"00": 70, "11": 30}:
    raise AssertionError("mock result mapping changed the configured counts")
print("PASS: mock transport/job/result contract succeeded; semantics not evaluated.")
