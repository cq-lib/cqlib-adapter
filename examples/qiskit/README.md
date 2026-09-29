# Qiskit examples

- 01_conversion.py: Qiskit -> real cqlib -> QCIS, no network.
- 02_mock_closed_loop.py: no-network Backend/Job/Result contract test. Its
  counts are preconfigured, so it exercises transport, task and result-shape
  contracts rather than quantum semantics.
- 03_tianyan_cloud.py: explicitly authorized real Tianyan submission.
- 031_tianyan_topology.py: Qiskit Target/cqlib topology and explicit physical-layout validation, followed by an optional real Tianyan run.

Run scripts from the repository root. See `docs/testing.md` for the shared test
environment and credential-safety rules.
