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
        "qiskit>=2.5,<3",
        "cirq-core>=1.7,<2",
        "pennylane>=0.45,<0.46",
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
    assert extras["cudaq"] == ["cudaq>=0.15,<0.16; platform_system != 'Windows'"]


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


def test_native_development_revisions_are_pinned_everywhere() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    native = data["tool"]["cqlib_adapter"]["native_dependencies"]
    references = {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/testing.md": (ROOT / "docs" / "testing.md").read_text(encoding="utf-8"),
        "ci.yml": (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        "cudaq.yml": (ROOT / ".github" / "workflows" / "cudaq.yml").read_text(encoding="utf-8"),
    }

    for dependency in native.values():
        revision = dependency["revision"]
        assert isinstance(revision, str)
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)
        assert dependency["package_version"] == "0.1.0"
        for text in references.values():
            assert revision in text

    for dependency_name, dependency in native.items():
        checkout = (
            f"git -C ../{dependency_name.replace('_', '-')} checkout {dependency['revision']}"
        )
        assert checkout in references["README.md"]
        assert checkout in references["docs/testing.md"]

    expected_checkouts = {"ci.yml": 4, "cudaq.yml": 1}
    for name, expected_count in expected_checkouts.items():
        workflow = references[name]
        for dependency_name, dependency in native.items():
            revision = dependency["revision"]
            checkout = f"repository: cq-lib/{dependency_name.replace('_', '-')}\n"
            assert workflow.count(checkout) == expected_count
            assert workflow.count(f"{checkout}          ref: {revision}\n") == expected_count
