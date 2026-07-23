"""Gate names accepted on the CUDA-Q OpenQASM 2 conversion path."""

from __future__ import annotations

from types import MappingProxyType

# CUDA-Q emits standard OpenQASM spellings and cqlib owns the actual parser.
# This table documents the stable M5 surface and is useful for diagnostics.
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
