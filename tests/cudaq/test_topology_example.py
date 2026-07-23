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

import runpy
from pathlib import Path

import pytest

cudaq = pytest.importorskip("cudaq")
pytest.importorskip("cqlib")
from cqlib.device import Layout  # noqa: E402

from cqlib_adapter.common import CompilationOptions  # noqa: E402
from cqlib_adapter.cudaq import CqlibSimulator, compile_cudaq_kernel  # noqa: E402

pytestmark = pytest.mark.cudaq


def test_topology_example_maps_cudaq_qubits_to_one_physical_path() -> None:
    source = runpy.run_path(
        str(Path(__file__).parents[2] / "examples" / "cudaq" / "031_tianyan_topology.py")
    )
    select_path = source["select_three_qubit_path"]
    assert_mapping = source["assert_compiled_mapping"]
    kernel = source["topology_kernel"]
    simulator = CqlibSimulator(4, seed=43)
    device = simulator.device
    path = select_path(device)
    layout = Layout.from_pairs(
        [(logical, physical) for logical, physical in enumerate(path)],
        physical_count=device.num_qubits,
    )

    artifact = compile_cudaq_kernel(
        kernel,
        device=device,
        options=CompilationOptions(initial_layout=layout, seed=43),
    )
    assert_mapping(artifact, device, path)

    assert tuple(item.physical_qubit for item in artifact.measurements) == path
    used_edges = {
        frozenset(qubit.index for qubit in operation.qubits)
        for operation in artifact.circuit.operations
        if len(operation.qubits) == 2
    }
    expected_edges = {frozenset(path[:2]), frozenset(path[1:])}
    assert used_edges
    assert used_edges <= expected_edges
