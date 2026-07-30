# Examples

Runnable framework examples are grouped by adapter:

- `qiskit/`: conversion, mock, simulator algorithms, real Tianyan and topology/layout.
- `pennylane/`: conversion, mock QNode, Grover, 011 bit order, real Tianyan and topology/layout.
- `cirq/`: conversion, mock Sampler, Grover, Deutsch-Jozsa, 011 measurement keys, real Tianyan and topology/layout.
- `cudaq/`: conversion, mock sync/async execution, cqlib simulator, Grover, 011 bit order, real Tianyan and topology/layout.

Every mock example tests transport, task and result-shape contracts with
preconfigured counts; it is not evidence of quantum-circuit semantics. The
local cqlib algorithm, statevector, bit-order and basis-measurement examples
provide the semantic checks.

Real-cloud examples are always explicit, contain only a blank API-key constant,
and never run as part of the default test suite.
