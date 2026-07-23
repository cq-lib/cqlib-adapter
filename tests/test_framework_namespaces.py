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

from importlib import import_module
from types import ModuleType

import pytest

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_namespace_availability_delegates_to_optional_discovery(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
) -> None:
    namespace = import_module(f"cqlib_adapter.{framework}")
    observed: list[str] = []

    def discover(module_name: str) -> bool:
        observed.append(module_name)
        return True

    monkeypatch.setattr(namespace, "is_dependency_available", discover)

    assert namespace.is_available()
    assert observed == [framework]


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_namespace_require_uses_its_own_extra(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
) -> None:
    namespace = import_module(f"cqlib_adapter.{framework}")
    sentinel = ModuleType(framework)
    observed: list[tuple[str, str, str | None]] = []

    def require(module_name: str, *, extra: str, display_name: str | None = None) -> ModuleType:
        observed.append((module_name, extra, display_name))
        return sentinel

    monkeypatch.setattr(namespace, "require_dependency", require)

    assert namespace.require_framework() is sentinel
    assert observed[0][0:2] == (framework, framework)


def test_common_namespace_is_importable_without_frameworks() -> None:
    common = import_module("cqlib_adapter.common")
    assert "CircuitCompiler" in common.__all__
    assert "TianyanConnector" in common.__all__
    assert "ResultConverter" in common.__all__
