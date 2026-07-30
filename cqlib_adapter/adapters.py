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

"""Static registry describing independently installable adapter namespaces."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from ._optional import is_dependency_available


@dataclass(frozen=True, slots=True)
class AdapterInfo:
    """Packaging metadata for one framework adapter."""

    name: str
    extra: str
    framework_module: str
    adapter_module: str

    @property
    def available(self) -> bool:
        """Return whether the framework dependency is discoverable."""

        return is_dependency_available(self.framework_module)


_ADAPTERS: Final = {
    "qiskit": AdapterInfo(
        name="Qiskit",
        extra="qiskit",
        framework_module="qiskit",
        adapter_module="cqlib_adapter.qiskit",
    ),
    "cirq": AdapterInfo(
        name="Cirq",
        extra="cirq",
        framework_module="cirq",
        adapter_module="cqlib_adapter.cirq",
    ),
    "pennylane": AdapterInfo(
        name="PennyLane",
        extra="pennylane",
        framework_module="pennylane",
        adapter_module="cqlib_adapter.pennylane",
    ),
    "cudaq": AdapterInfo(
        name="CUDA-Q",
        extra="cudaq",
        framework_module="cudaq",
        adapter_module="cqlib_adapter.cudaq",
    ),
}

ADAPTERS = MappingProxyType(_ADAPTERS)


def get_adapter_info(name: str) -> AdapterInfo:
    """Return adapter metadata by normalized framework name."""

    key = name.casefold().replace("-", "")
    aliases = {"cudaq": "cudaq", "pennylane": "pennylane", "qiskit": "qiskit", "cirq": "cirq"}
    try:
        return ADAPTERS[aliases[key]]
    except KeyError as exc:
        supported = ", ".join(ADAPTERS)
        raise KeyError(f"Unknown adapter {name!r}. Supported adapters: {supported}") from exc


def available_adapters() -> tuple[str, ...]:
    """Return installed framework adapter names without importing the frameworks."""

    return tuple(name for name, info in ADAPTERS.items() if info.available)
