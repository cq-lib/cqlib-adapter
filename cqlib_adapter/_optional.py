"""Helpers for optional framework dependencies."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import ModuleType


class OptionalDependencyError(ImportError):
    """Raised when a requested framework extra is not installed or cannot be imported."""


def is_dependency_available(module_name: str) -> bool:
    """Return whether an optional top-level module can be discovered without importing it."""

    try:
        return find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def require_dependency(
    module_name: str,
    *,
    extra: str,
    display_name: str | None = None,
) -> ModuleType:
    """Import an optional framework or raise an actionable adapter-specific error."""

    label = display_name or module_name
    if not is_dependency_available(module_name):
        raise OptionalDependencyError(
            f"{label} is required for this adapter. "
            f'Install it with: pip install "cqlib-adapter[{extra}]"'
        )
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise OptionalDependencyError(
            f"{label} was found but could not be imported. Check the framework's platform "
            f'and binary requirements, then reinstall "cqlib-adapter[{extra}]".'
        ) from exc
