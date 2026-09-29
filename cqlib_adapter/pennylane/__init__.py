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

"""PennyLane adapter with lazy optional-dependency loading."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Final

from cqlib_adapter._optional import is_dependency_available, require_dependency

EXTRA_NAME: Final = "pennylane"
FRAMEWORK_MODULE: Final = "pennylane"

_EXPORTS: Final = {
    "FSim": (".operations", "FSim"),
    "FSimGate": (".operations", "FSimGate"),
    "RXY": (".operations", "RXY"),
    "RXYGate": (".operations", "RXYGate"),
    "X2M": (".operations", "X2M"),
    "X2MGate": (".operations", "X2MGate"),
    "X2P": (".operations", "X2P"),
    "X2PGate": (".operations", "X2PGate"),
    "XY": (".operations", "XY"),
    "XY2M": (".operations", "XY2M"),
    "XY2MGate": (".operations", "XY2MGate"),
    "XY2P": (".operations", "XY2P"),
    "XY2PGate": (".operations", "XY2PGate"),
    "XYGate": (".operations", "XYGate"),
    "Y2M": (".operations", "Y2M"),
    "Y2MGate": (".operations", "Y2MGate"),
    "Y2P": (".operations", "Y2P"),
    "Y2PGate": (".operations", "Y2PGate"),
    "pennylane_to_cqlib": (".converter", "pennylane_to_cqlib"),
    "cqlib_to_pennylane": (".converter", "cqlib_to_pennylane"),
    "compile_pennylane_circuit": (".converter", "compile_pennylane_circuit"),
    "canonical_to_pennylane_result": (".result", "canonical_to_pennylane_result"),
    "PennyLaneExecution": (".execution", "PennyLaneExecution"),
    "TianyanDevice": (".device", "TianyanDevice"),
}


def is_available() -> bool:
    """Return whether PennyLane is installed without importing it."""

    return is_dependency_available(FRAMEWORK_MODULE)


def require_framework() -> ModuleType:
    """Import PennyLane or raise an actionable optional-dependency error."""

    return require_dependency(FRAMEWORK_MODULE, extra=EXTRA_NAME, display_name="PennyLane")


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    require_framework()
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "EXTRA_NAME",
    "FRAMEWORK_MODULE",
    "RXY",
    "X2M",
    "X2P",
    "XY",
    "XY2M",
    "XY2P",
    "Y2M",
    "Y2P",
    "FSim",
    "FSimGate",
    "PennyLaneExecution",
    "RXYGate",
    "TianyanDevice",
    "X2MGate",
    "X2PGate",
    "XY2MGate",
    "XY2PGate",
    "XYGate",
    "Y2MGate",
    "Y2PGate",
    "canonical_to_pennylane_result",
    "compile_pennylane_circuit",
    "cqlib_to_pennylane",
    "is_available",
    "pennylane_to_cqlib",
    "require_framework",
]
