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

"""Cirq adapter with lazy optional-dependency loading."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Final

from cqlib_adapter._optional import is_dependency_available, require_dependency

EXTRA_NAME: Final = "cirq"
FRAMEWORK_MODULE: Final = "cirq"

_EXPORTS: Final = {
    "canonical_to_cirq_probabilities": (
        ".result",
        "canonical_to_cirq_probabilities",
    ),
    "CirqExecution": (".job", "CirqExecution"),
    "CqlibSimulatorSampler": (".local_simulator", "CqlibSimulatorSampler"),
    "CirqStatevectorResult": (".local_simulator", "CirqStatevectorResult"),
    "FSimGate": (".gates", "FSimGate"),
    "QCIS_GATE_TYPES": (".gates", "QCIS_GATE_TYPES"),
    "RXYGate": (".gates", "RXYGate"),
    "TianyanDevice": (".device", "TianyanDevice"),
    "TianyanSampler": (".sampler", "TianyanSampler"),
    "X2MGate": (".gates", "X2MGate"),
    "X2PGate": (".gates", "X2PGate"),
    "XY2MGate": (".gates", "XY2MGate"),
    "XY2PGate": (".gates", "XY2PGate"),
    "XYGate": (".gates", "XYGate"),
    "Y2MGate": (".gates", "Y2MGate"),
    "Y2PGate": (".gates", "Y2PGate"),
    "canonical_to_cirq_result": (".result", "canonical_to_cirq_result"),
    "cirq_to_cqlib": (".converter", "cirq_to_cqlib"),
    "compile_cirq_circuit": (".converter", "compile_cirq_circuit"),
    "cqlib_to_cirq": (".converter", "cqlib_to_cirq"),
    "decompose_cirq_circuit": (".converter", "decompose_cirq_circuit"),
    "qcis_gate": (".gates", "qcis_gate"),
}


def is_available() -> bool:
    return is_dependency_available(FRAMEWORK_MODULE)


def require_framework() -> ModuleType:
    return require_dependency(FRAMEWORK_MODULE, extra=EXTRA_NAME, display_name="Cirq")


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
    "QCIS_GATE_TYPES",
    "CirqExecution",
    "CirqStatevectorResult",
    "CqlibSimulatorSampler",
    "FSimGate",
    "RXYGate",
    "TianyanDevice",
    "TianyanSampler",
    "X2MGate",
    "X2PGate",
    "XY2MGate",
    "XY2PGate",
    "XYGate",
    "Y2MGate",
    "Y2PGate",
    "canonical_to_cirq_probabilities",
    "canonical_to_cirq_result",
    "cirq_to_cqlib",
    "compile_cirq_circuit",
    "cqlib_to_cirq",
    "decompose_cirq_circuit",
    "is_available",
    "qcis_gate",
    "require_framework",
]
