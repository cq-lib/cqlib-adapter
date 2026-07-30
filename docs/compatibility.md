# Compatibility baseline

Baseline recorded on 2026-07-20:

| Component | Baseline constraint | Notes |
|---|---:|---|
| Python | `>=3.11` | Common denominator for the four current framework releases. |
| cqlib | `==0.1.0` | New local Rust/PyO3 implementation; do not substitute legacy PyPI 1.x. |
| cqlib-tianyan | `==0.1.0` | New local Tianyan client. |
| Qiskit | `>=2.5,<3` | Executable compatibility tests cover BackendV2, Target, Job, Result and Sampler. |
| Cirq Core | `>=1.7,<2` | Executable tests target Circuit, DeviceMetadata, Sampler/run_sweep and ResultDict in 1.7.0. |
| PennyLane | `>=0.45,<0.46` | Executable tests target the current `Device`/`QNode` execution contract in 0.45.1. |
| CUDA-Q | `>=0.15,<0.16` | Linux/macOS only; use WSL2 from Windows. |

The common suite verifies cqlib compile/QCIS/device/result contracts and the cqlib-tianyan backend/task contracts. Framework suites provide executable compatibility tests for Qiskit, PennyLane, Cirq and CUDA-Q. CUDA-Q is verified independently on Linux/WSL with the `qpp-cpu` target.
