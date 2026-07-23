from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

from cqlib.device import Layout

from cqlib_adapter.common import CompilationOptions, NormalizedDevice
from cqlib_adapter.pennylane import compile_pennylane_circuit
from cqlib_adapter.pennylane.testing import MockCloudBackend


def load_example() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return runpy.run_path(str(root / "examples" / "pennylane" / "031_tianyan_topology.py"))


def test_topology_example_maps_pennylane_wires_to_one_physical_path() -> None:
    source = load_example()
    device = NormalizedDevice.from_backend(MockCloudBackend([], size=5))
    path = source["select_three_qubit_path"](device)
    assert path == (0, 1, 2)
    layout = Layout.from_pairs(
        [(logical, physical) for logical, physical in enumerate(path)],
        physical_count=device.num_qubits,
    )
    artifact = compile_pennylane_circuit(
        source["topology_tape"](20),
        wire_order=(0, 1, 2),
        device=device,
        options=CompilationOptions(initial_layout=layout, seed=41),
    )

    source["assert_compiled_mapping"](artifact, device, path)
    assert tuple(item.physical_qubit for item in artifact.measurements) == path
    assert "CZ Q0 Q1" in artifact.qcis
    assert "CZ Q1 Q2" in artifact.qcis
