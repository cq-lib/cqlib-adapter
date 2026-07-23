from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist() -> None:
    required = {
        "README.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "pyproject.toml",
        "environment-dev.yml",
        "requirements-dev.txt",
        "MANIFEST.in",
        ".gitignore",
        ".github/workflows/ci.yml",
    }
    missing = sorted(path for path in required if not (ROOT / path).is_file())
    assert not missing, f"Missing project files: {missing}"


def test_framework_package_boundaries_exist() -> None:
    package = ROOT / "cqlib_adapter"
    for namespace in ("common", "qiskit", "cirq", "pennylane", "cudaq"):
        assert (package / namespace / "__init__.py").is_file()


def test_test_sources_are_not_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    ignored_lines = {
        line.strip()
        for line in gitignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "tests/" not in ignored_lines
    assert "test_*.py" not in ignored_lines
