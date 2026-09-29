# Examples

Runnable framework examples are grouped by adapter:

- `qiskit/`: conversion, mock closed loop, real Tianyan and topology/layout.
- `pennylane/`: conversion, mock QNode, real Tianyan and topology/layout.
- `cirq/`: conversion, mock Sampler, real Tianyan and topology/layout.
- `cudaq/`: conversion, mock sync/async execution, real Tianyan and topology/layout.

Every mock example tests transport, task and result-shape contracts with
preconfigured counts; it is not evidence of quantum-circuit semantics. Mock
closed-loop runs are the only offline execution path.

Real-cloud examples are always explicit, contain only a blank API-key constant,
and never run as part of the default test suite.
