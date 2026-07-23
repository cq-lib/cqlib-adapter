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

"""Gate names accepted by the optional CUDA-Q OpenQASM 2 diagnostic exporter."""

from __future__ import annotations

from types import MappingProxyType

# This table documents the diagnostic export surface; production conversion
# uses Quake MLIR objects and does not consult it.
OPENQASM_TO_CQLIB = MappingProxyType(
    {
        "id": "I",
        "h": "H",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "s": "S",
        "sdg": "SDG",
        "t": "T",
        "tdg": "TDG",
        "rx": "RX",
        "ry": "RY",
        "rz": "RZ",
        "u1": "PHASE",
        "u2": "U",
        "u3": "U",
        "cx": "CX",
        "cy": "CY",
        "cz": "CZ",
        "swap": "SWAP",
        "ccx": "CCX",
        "reset": "RESET",
        "barrier": "BARRIER",
        "measure": "MEASURE",
    }
)

SUPPORTED_OPENQASM_GATES = frozenset(OPENQASM_TO_CQLIB)


def cqlib_gate_for_openqasm(name: str) -> str:
    """Normalize one documented OpenQASM gate name to cqlib spelling."""

    try:
        return OPENQASM_TO_CQLIB[name.strip().lower()]
    except KeyError as exc:
        raise KeyError(f"unsupported CUDA-Q OpenQASM gate {name!r}") from exc


__all__ = ["OPENQASM_TO_CQLIB", "SUPPORTED_OPENQASM_GATES", "cqlib_gate_for_openqasm"]
