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

from types import ModuleType

import pytest

from cqlib_adapter import _optional
from cqlib_adapter._optional import OptionalDependencyError


def test_missing_dependency_error_contains_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_optional, "find_spec", lambda _name: None)

    with pytest.raises(OptionalDependencyError, match=r"cqlib-adapter\[qiskit\]"):
        _optional.require_dependency("qiskit", extra="qiskit", display_name="Qiskit")


def test_broken_binary_import_is_wrapped_and_chained(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(_optional, "find_spec", lambda _name: sentinel)

    original = ImportError("missing native library")

    def fail_import(_name: str) -> ModuleType:
        raise original

    monkeypatch.setattr(_optional, "import_module", fail_import)

    with pytest.raises(OptionalDependencyError) as captured:
        _optional.require_dependency("cudaq", extra="cudaq", display_name="CUDA-Q")

    assert captured.value.__cause__ is original


@pytest.mark.parametrize("exception", [ImportError, ModuleNotFoundError, ValueError])
def test_discovery_errors_are_treated_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    exception: type[Exception],
) -> None:
    def fail_spec(_name: str) -> None:
        raise exception("bad module state")

    monkeypatch.setattr(_optional, "find_spec", fail_spec)
    assert not _optional.is_dependency_available("framework")
