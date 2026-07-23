# Qiskit examples

- 01_conversion.py: Qiskit -> real cqlib -> QCIS, no network.
- 02_mock_closed_loop.py: no-network Backend/Job/Result contract test. Its
  counts are preconfigured, so quantum semantics are verified by the local
  cqlib examples instead.
- 03_tianyan_cloud.py: explicitly authorized real Tianyan submission.
- 031_tianyan_topology.py: Qiskit Target/cqlib topology and explicit physical-layout validation, followed by an optional real Tianyan run.
- 04_grover_simulator.py: Qiskit vs local cqlib Grover simulation.
- 05_deutsch_jozsa_simulator.py: Qiskit vs local cqlib DJ simulation.
- 06_scaling_simulator.py: logged Qiskit BasicSimulator vs Qiskit -> cqlib ->
  native QCIS -> cqlib simulator comparison over multiple qubit counts/depths.
  Start with: python examples/qiskit/06_scaling_simulator.py --smoke
- 08_basis_measurement.py: deterministic X/Y-basis rotations through the real
  local cqlib compilation and simulation path.

Run scripts from the repository root. See docs/m2-qiskit-testing.md first.
