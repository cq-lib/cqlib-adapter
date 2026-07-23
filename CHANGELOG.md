# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added

- Package skeleton for shared, Qiskit, Cirq, PennyLane, and CUDA-Q namespaces.
- Independent optional dependency groups for the four frameworks.
- Development environment, test configuration, linting, typing, build, and CI foundations.
- Framework-neutral circuit/measurement contracts and validated run/compile options.
- cqlib compiler boundary with QCIS round-trip, native-basis, topology and measurement checks.
- Tianyan device normalization, authentication/discovery connector and calibration-aware submission.
- Ordered Job status/wait/timeout/cache handling and canonical counts/probabilities/samples conversion.
- Offline common-core fakes, boundary tests and a complete compile-to-result closed-loop test.
- Qiskit QCIS-native gates and Qiskit/cqlib bidirectional circuit conversion.
- Qiskit Target/CouplingMap generation with native-gate decomposition and topology routing.
- Tianyan BackendV2, JobV1, standard Result and BackendSamplerV2 surfaces.
- Real-cqlib offline cloud doubles, opt-in Tianyan live test and three runnable Qiskit examples.
- PennyLane Device/QNode, native operations, counts/probs/sample conversion, local cqlib simulation, mock/live tests and examples.
- Cirq Circuit conversion, QCIS gates, DeviceMetadata, Sampler/run_sweep, measurement-key ResultDict conversion, local cqlib simulation, mock/live topology tests and examples.
- CUDA-Q kernel/OpenQASM 2 conversion, automatic measurement, QCIS compilation, target metadata, sync/async execution, compatible sample results, local simulation, Linux CI and opt-in live-cloud validation.
- Cross-framework exact statevector scaling checks, routed logical-layout restoration and basis-measurement examples.

### Security

- Keep all committed API-key placeholders empty and read live credentials only from the current process environment.
- Redact Tianyan login secrets from provider exception messages and suppress the original exception chain.
- Require explicit cloud-test authorization and disable credential persistence in every live example.
