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

import tomllib
from pathlib import Path

import cqlib_adapter

ROOT = Path(__file__).resolve().parents[1]


def load_pyproject() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def test_version_identifies_second_generation() -> None:
    assert cqlib_adapter.__version__ == "2.0.0.dev0"


def test_python_baseline_is_shared_by_current_frameworks() -> None:
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    assert project["requires-python"] == ">=3.11"


def test_base_dependencies_are_pinned_to_new_local_product_line() -> None:
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    assert project["dependencies"] == ["cqlib==0.1.0", "cqlib-tianyan==0.1.0"]


def test_framework_extras_are_independent_and_all_is_their_union() -> None:
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)
    assert {"qiskit", "cirq", "pennylane", "cudaq", "all", "dev"} <= set(extras)

    individual = {
        dependency
        for name in ("qiskit", "cirq", "pennylane", "cudaq")
        for dependency in extras[name]
    }
    assert set(extras["all"]) == individual


def test_cudaq_extra_is_guarded_on_native_windows() -> None:
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)
    assert extras["cudaq"] == ["cudaq>=0.15,<0.16; platform_system != 'Windows'"]
