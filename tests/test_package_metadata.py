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
from configparser import ConfigParser
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import cqlib_adapter

ROOT = Path(__file__).resolve().parents[1]


def load_pyproject() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def test_version_is_valid_pep440() -> None:
    Version(cqlib_adapter.__version__)


def test_version_matches_installed_distribution() -> None:
    try:
        installed = distribution_version("cqlib-adapter")
    except PackageNotFoundError:
        pytest.skip("cqlib-adapter is not installed")
    assert Version(cqlib_adapter.__version__) == Version(installed)


def test_python_baseline_is_shared_by_current_frameworks() -> None:
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    assert project["requires-python"] == ">=3.11"


def test_base_dependencies_declare_bounded_compatible_ranges() -> None:
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    dependencies = [Requirement(item) for item in project["dependencies"]]
    assert {requirement.name for requirement in dependencies} == {"cqlib", "cqlib-tianyan"}
    for requirement in dependencies:
        operators = {specifier.operator for specifier in requirement.specifier}
        assert "==" not in operators
        assert ">=" in operators
        assert operators & {"<", "<="}


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


def test_dev_extra_contains_the_windows_supported_framework_test_surface() -> None:
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)
    dev = set(extras["dev"])
    for name in ("qiskit", "cirq", "pennylane"):
        assert set(extras[name]) <= dev


def test_dev_convenience_files_match_the_windows_framework_surface() -> None:
    expected = {
        "qiskit>=2.1,<3",
        "cirq-core>=1.4,<2",
        "pennylane>=0.44,<1",
    }
    requirements = set((ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines())
    environment = (ROOT / "environment-dev.yml").read_text(encoding="utf-8")

    assert expected <= requirements
    for dependency in expected:
        assert f"      - {dependency}" in environment


def test_cudaq_extra_is_guarded_on_native_windows() -> None:
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)
    assert extras["cudaq"] == ["cudaq>=0.15,<0.17; platform_system != 'Windows'"]


def test_windows_coverage_config_excludes_only_the_unavailable_cudaq_surface() -> None:
    config = ConfigParser()
    loaded = config.read(ROOT / "coverage-windows.ini", encoding="utf-8")

    assert loaded == [str(ROOT / "coverage-windows.ini")]
    assert config["run"]["source"].strip() == "cqlib_adapter"
    assert config["report"].getint("fail_under") == 80
    assert config["report"]["omit"].strip() == "cqlib_adapter/cudaq/*"


def test_docs_and_ci_exercise_the_platform_specific_coverage_commands() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    testing = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    cudaq_ci = (ROOT / ".github" / "workflows" / "cudaq.yml").read_text(encoding="utf-8")

    windows_coverage = "--cov-config=coverage-windows.ini --cov-report=term-missing"
    linux_coverage = "--cov=cqlib_adapter --cov-report=term-missing"
    assert windows_coverage in readme
    assert windows_coverage in testing
    assert windows_coverage in ci
    assert linux_coverage in readme
    assert linux_coverage in testing
    assert linux_coverage in cudaq_ci
    assert 'python -m pip install -e ".[dev,cudaq]"' in cudaq_ci
