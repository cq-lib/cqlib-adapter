"""CUDA-Q adapter with lazy optional-dependency loading."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Final

from cqlib_adapter._optional import is_dependency_available, require_dependency

EXTRA_NAME: Final = "cudaq"
FRAMEWORK_MODULE: Final = "cudaq"

_EXPORTS: Final = {
    "CqlibSimulator": (".local_simulator", "CqlibSimulator"),
    "CudaQStatevectorResult": (".local_simulator", "CudaQStatevectorResult"),
    "CudaQJob": (".job", "CudaQJob"),
    "CudaQSampleResult": (".result", "CudaQSampleResult"),
    "TianyanExecutor": (".execution", "TianyanExecutor"),
    "TianyanTarget": (".target", "TianyanTarget"),
    "canonical_to_cudaq_result": (".result", "canonical_to_cudaq_result"),
    "compile_cudaq_kernel": (".converter", "compile_cudaq_kernel"),
    "cudaq_to_cqlib": (".converter", "cudaq_to_cqlib"),
    "cudaq_to_openqasm": (".converter", "cudaq_to_openqasm"),
    "target_from_device": (".target", "target_from_device"),
}

_FRAMEWORK_FREE_EXPORTS: Final = frozenset(
    {
        "CudaQSampleResult",
        "TianyanTarget",
        "canonical_to_cudaq_result",
        "target_from_device",
    }
)


def is_available() -> bool:
    """Return whether CUDA-Q is installed without importing it."""

    return is_dependency_available(FRAMEWORK_MODULE)


def require_framework() -> ModuleType:
    """Import CUDA-Q or raise an actionable optional-dependency error."""

    return require_dependency(FRAMEWORK_MODULE, extra=EXTRA_NAME, display_name="CUDA-Q")


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    if name not in _FRAMEWORK_FREE_EXPORTS:
        require_framework()
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "EXTRA_NAME",
    "FRAMEWORK_MODULE",
    "CqlibSimulator",
    "CudaQJob",
    "CudaQSampleResult",
    "CudaQStatevectorResult",
    "TianyanExecutor",
    "TianyanTarget",
    "canonical_to_cudaq_result",
    "compile_cudaq_kernel",
    "cudaq_to_cqlib",
    "cudaq_to_openqasm",
    "is_available",
    "require_framework",
    "target_from_device",
]
