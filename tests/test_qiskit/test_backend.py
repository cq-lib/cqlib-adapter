# This code is part of cqlib.
#
# Copyright (C) 2025 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum backend integration test module.

This module validates the integration with the TianYan quantum computing service provider.
Tests cover authentication, backend discovery, job submission, and result retrieval for both:
- Physical quantum devices
- Quantum simulators

Verification includes:
- Provider authentication and backend enumeration
- Circuit transpilation compatibility
- Job execution lifecycle management
- Basic quantum operation correctness
"""

import os

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

from cqlib_adapter.qiskit_ext.tianyan_provider import TianYanProvider


class TestBackend:
    """Main test class for TianYan provider backend integration.

    Attributes:
        _provider (TianYanProvider): Authenticated provider instance shared across tests
    """

    @classmethod
    def setup_class(cls):
        """Initializes provider with environment token."""
        cls._provider = TianYanProvider(token=os.environ.get("CQLIB_TOKEN"))
        assert cls._provider.token is not None

    def test_backends(self):
        """Validates backend discovery and filtering capabilities.

        Test cases:
        1. Filters physical devices (simulator=False)
        2. Filters online-available backends (online=True)

        Verifies:
        - Correct backend type filtering
        - Valid online_date attribute for online backends
        """
        backends = self._provider.backends(simulator=False)
        for backend in backends:
            assert backend.simulator is False
        backends = self._provider.backends(online=True)
        for backend in backends:
            assert backend.configuration.online_date is not None

    def test_run(self):
        """End-to-end quantum job execution test on physical device.

        Test workflow:
        1. Constructs 3-qubit circuit with mixed gates:
           - X gate
           - Hadamard
           - Toffoli (CCX)
           - Controlled-RX
        2. Transpiles for target backend (tianyan24)
        3. Submits 3000-shot job
        4. Validates:
           - Successful job ID generation
           - Measurement result retrieval

        Uses real quantum device backend.
        """
        backend = self._provider.backend('tianyan24')
        qs = QuantumRegister(3)
        cs = ClassicalRegister(3)
        circuit = QuantumCircuit(qs, cs)
        circuit.x(qs[0])
        circuit.h(qs[1])
        circuit.ccx(qs[0], qs[1], qs[2])
        circuit.crx(0.5, qs[1], qs[2])
        circuit.barrier(qs)
        circuit.measure(qs, cs)

        transpiled_qc = transpile(circuit, backend=backend)
        job = backend.run([transpiled_qc], shots=3000)
        assert job.job_id()
        assert job.result().get_counts()

    def test_sim(self):
        """Quantum simulator workflow validation.

        Tests basic simulator capabilities:
        1. Builds 3-qubit circuit with:
           - Toffoli gate
           - Controlled-RX(0.5)
        2. Transpiles for simulator backend (tianyan_sw)
        3. Executes 3000-shot simulation
        4. Verifies:
           - Valid job completion
           - Measurement result existence

        Focuses on simulator-specific execution path.
        """
        backend = self._provider.backend('tianyan_sw')
        qs = QuantumRegister(3)
        cs = ClassicalRegister(3)
        circuit = QuantumCircuit(qs, cs)
        circuit.ccx(qs[0], qs[1], qs[2])
        circuit.crx(0.5, qs[1], qs[2])
        circuit.measure(qs, cs)

        transpiled_qc = transpile(circuit, backend=backend)
        job = backend.run([transpiled_qc], shots=3000)
        assert job.job_id()
        assert job.result().get_counts()
