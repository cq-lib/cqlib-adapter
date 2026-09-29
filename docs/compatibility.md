# Compatibility baseline

Baseline recorded on 2026-07-20:

| Component | Baseline constraint | Notes |
|---|---:|---|
| Python | `>=3.11` | Common denominator for the four current framework releases. |
| cqlib | Declared in `pyproject.toml` | Published on PyPI with prebuilt wheels; do not substitute legacy PyPI 1.x. |
| cqlib-tianyan | Declared in `pyproject.toml` | New Tianyan client published on PyPI. |
| Qiskit | `>=2.1,<3` | Executable compatibility tests cover BackendV2, Target, Job, Result and Sampler; CI exercises the floor and the latest release. |
| Cirq Core | `>=1.4,<2` | Executable tests cover Circuit, DeviceMetadata, Sampler/run_sweep and ResultDict; CI exercises the floor and the latest release. |
| PennyLane | `>=0.44,<1` | Executable tests cover the current `Device`/`QNode` execution contract; CI exercises the floor and the latest release. |
| CUDA-Q | `>=0.15,<0.17` | Linux/macOS only; use WSL2 from Windows. CI exercises the floor and the latest release. |

The common suite verifies cqlib compile/QCIS/device/result contracts and the cqlib-tianyan backend/task contracts. Framework suites provide executable compatibility tests for Qiskit, PennyLane, Cirq and CUDA-Q. CUDA-Q is verified independently on Linux/WSL with the `qpp-cpu` target.
