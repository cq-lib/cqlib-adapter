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

from __future__ import annotations

import pytest

cudaq = pytest.importorskip("cudaq")
pytest.importorskip("cqlib")

from cqlib_adapter.common import CompilationOptions  # noqa: E402
from cqlib_adapter.cudaq import (  # noqa: E402
    CqlibSimulator,
    compile_cudaq_kernel,
    cudaq_to_cqlib,
)

pytestmark = [pytest.mark.integration, pytest.mark.cudaq]

NATIVE_BASIS = (
    "RZ",
    "X2P",
    "X2M",
    "Y2P",
    "Y2M",
    "XY2P",
    "XY2M",
    "CZ",
    "GPHASE",
)


@cudaq.kernel
def bell_without_measurement() -> None:
    qubits = cudaq.qvector(2)
    h(qubits[0])  # noqa: F821
    x.ctrl(qubits[0], qubits[1])  # noqa: F821


@cudaq.kernel
def state_110() -> None:
    qubits = cudaq.qvector(3)
    x(qubits[0])  # noqa: F821
    x(qubits[1])  # noqa: F821
    mz(qubits)  # noqa: F821


def test_cudaq_converter_returns_rust_backed_cqlib_circuit() -> None:
    from cqlib import Circuit

    bundle = cudaq_to_cqlib(bell_without_measurement)

    assert isinstance(bundle.circuit, Circuit)
    assert bundle.metadata.extras["auto_measure_all"] is True
    assert bundle.metadata.measurements.num_classical_bits == 2


def test_cudaq_compiler_calls_real_cqlib_compile_and_qcis() -> None:
    artifact = compile_cudaq_kernel(
        bell_without_measurement,
        options=CompilationOptions(target_basis=NATIVE_BASIS, seed=29),
    )

    assert "CZ Q0 Q1" in artifact.qcis
    assert artifact.qcis.count("M Q") == 2
    assert len(artifact.measurements) == 2


def test_cudaq_110_runs_through_real_cqlib_statevector_and_result() -> None:
    simulator = CqlibSimulator(3, seed=31)

    result = simulator.sample(state_110, shots_count=32)

    assert dict(result) == {"110": 32}
    assert simulator.last_qcis is not None
    assert simulator.last_qcis.count("M Q") == 3
    assert simulator.simulator_calls[0][2] == 32
