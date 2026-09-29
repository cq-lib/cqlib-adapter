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

"""Qiskit adapter with lazy optional-dependency loading."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Final

from cqlib_adapter._optional import is_dependency_available, require_dependency

EXTRA_NAME: Final = "qiskit"
FRAMEWORK_MODULE: Final = "qiskit"

_EXPORTS: Final = {
    "FSimGate": (".gates", "FSimGate"),
    "RXYGate": (".gates", "RXYGate"),
    "X2MGate": (".gates", "X2MGate"),
    "X2PGate": (".gates", "X2PGate"),
    "XY2MGate": (".gates", "XY2MGate"),
    "XY2PGate": (".gates", "XY2PGate"),
    "XYGate": (".gates", "XYGate"),
    "Y2MGate": (".gates", "Y2MGate"),
    "Y2PGate": (".gates", "Y2PGate"),
    "qcis_instruction": (".gates", "qcis_instruction"),
    "qcis_name_mapping": (".gates", "qcis_name_mapping"),
    "qiskit_to_cqlib": (".converter", "qiskit_to_cqlib"),
    "cqlib_to_qiskit": (".converter", "cqlib_to_qiskit"),
    "compile_qiskit_circuit": (".converter", "compile_qiskit_circuit"),
    "coupling_map_from_device": (".target", "coupling_map_from_device"),
    "target_from_device": (".target", "target_from_device"),
    "TianyanBackend": (".backend", "TianyanBackend"),
    "TianyanBackendStatus": (".backend", "TianyanBackendStatus"),
    "TianyanJob": (".job", "TianyanJob"),
    "canonical_to_qiskit_result": (".result", "canonical_to_qiskit_result"),
    "TianyanSampler": (".sampler", "TianyanSampler"),
}


def is_available() -> bool:
    """Return whether Qiskit is installed without importing it."""

    return is_dependency_available(FRAMEWORK_MODULE)


def require_framework() -> ModuleType:
    """Import Qiskit or raise an actionable optional-dependency error."""

    return require_dependency(FRAMEWORK_MODULE, extra=EXTRA_NAME, display_name="Qiskit")


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
    "FSimGate",
    "RXYGate",
    "TianyanBackend",
    "TianyanBackendStatus",
    "TianyanJob",
    "TianyanSampler",
    "X2MGate",
    "X2PGate",
    "XY2MGate",
    "XY2PGate",
    "XYGate",
    "Y2MGate",
    "Y2PGate",
    "canonical_to_qiskit_result",
    "compile_qiskit_circuit",
    "coupling_map_from_device",
    "cqlib_to_qiskit",
    "is_available",
    "qcis_instruction",
    "qcis_name_mapping",
    "qiskit_to_cqlib",
    "require_framework",
    "target_from_device",
]
