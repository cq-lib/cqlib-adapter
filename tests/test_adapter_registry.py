from __future__ import annotations

import pytest

from cqlib_adapter import ADAPTERS, adapters, available_adapters, get_adapter_info


def test_registry_contains_exactly_four_frameworks() -> None:
    assert tuple(ADAPTERS) == ("qiskit", "cirq", "pennylane", "cudaq")


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("QISKIT", "qiskit"), ("Cirq", "cirq"), ("PennyLane", "pennylane"), ("CUDA-Q", "cudaq")],
)
def test_registry_normalizes_user_facing_names(requested: str, expected: str) -> None:
    assert get_adapter_info(requested) is ADAPTERS[expected]


def test_unknown_adapter_lists_supported_values() -> None:
    with pytest.raises(KeyError, match="qiskit, cirq, pennylane, cudaq"):
        get_adapter_info("unknown")


def test_available_adapters_uses_discovery_without_importing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapters,
        "is_dependency_available",
        lambda module: module in {"qiskit", "cirq"},
    )
    assert available_adapters() == ("qiskit", "cirq")


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        ADAPTERS["new"] = ADAPTERS["qiskit"]  # type: ignore[index]
