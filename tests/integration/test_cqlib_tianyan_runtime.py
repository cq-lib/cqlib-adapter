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

import inspect
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import requires, version
from pathlib import Path
from typing import Any

import cqlib_tianyan
import pytest
from cqlib_tianyan import (
    CalibrationMode,
    TaskHandle,
    TianyanBackend,
    TianyanConfig,
    TianyanPlatform,
)
from packaging.requirements import Requirement

from cqlib_adapter.common import AdapterSubmissionError, TianyanConnector

pytestmark = pytest.mark.integration


def test_uses_local_rust_compiled_cqlib_tianyan_extension() -> None:
    import cqlib_tianyan._cqlib_tianyan as native

    requirement = next(
        requirement
        for requirement in (Requirement(item) for item in requires("cqlib-adapter") or [])
        if requirement.name == "cqlib-tianyan"
    )
    assert requirement.specifier.contains(version("cqlib-tianyan"), prereleases=True)
    assert Path(cqlib_tianyan.__file__).resolve().is_file()
    native_path = str(Path(native.__file__).resolve())
    assert any(native_path.endswith(suffix) for suffix in EXTENSION_SUFFIXES)


def test_native_configuration_and_calibration_modes_are_python_compatible() -> None:
    config = TianyanConfig(
        domain="example.test",
        save_credentials=False,
        auto_refresh=False,
        credentials_path="D:/temporary/credentials.json",
    )
    assert config.domain == "example.test"
    assert config.base_url == "https://example.test"
    assert config.save_credentials is False
    assert config.auto_refresh is False
    assert config.credentials_path == "D:/temporary/credentials.json"

    for value in ("auto", "enabled", "disabled"):
        mode = CalibrationMode(value)
        assert mode.value == value
        assert str(mode) == value
    with pytest.raises(ValueError, match="CalibrationMode"):
        CalibrationMode("invalid")


def test_adapter_calls_are_supported_by_native_tianyan_signatures() -> None:
    inspect.signature(TianyanPlatform.login).bind(
        "",
        domain="example.test",
        save_credentials=False,
        auto_refresh=False,
        credentials_path="credentials.json",
    )
    inspect.signature(TianyanPlatform.from_credentials).bind(
        domain="example.test",
        save_credentials=False,
    )
    inspect.signature(TianyanPlatform.list_backends).bind(object())
    inspect.signature(TianyanPlatform.get_backend).bind(object(), "device")
    inspect.signature(TianyanBackend.run).bind(object(), ["M Q0"], 100)
    inspect.signature(TianyanBackend.run_raw).bind(object(), ["M Q0"], 100)
    inspect.signature(TianyanBackend.run_with_mode).bind(object(), ["M Q0"], 100, "enabled")
    inspect.signature(TianyanBackend.device_config).bind(object())
    inspect.signature(TaskHandle.status).bind(object())
    inspect.signature(TaskHandle.wait).bind(object(), timeout=120.0, poll_interval=5.0)


def test_connector_login_and_credentials_bridge_imported_native_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class PlatformDouble:
        @staticmethod
        def login(*args: Any, **kwargs: Any) -> PlatformDouble:
            calls.append(("login", args, kwargs))
            return PlatformDouble()

        @staticmethod
        def from_credentials(**kwargs: Any) -> PlatformDouble:
            calls.append(("credentials", (), kwargs))
            return PlatformDouble()

        def list_backends(self) -> list[object]:
            return []

        def get_backend(self, name: str) -> object:
            raise KeyError(name)

    monkeypatch.setattr(cqlib_tianyan, "TianyanPlatform", PlatformDouble)

    login_connector = TianyanConnector.login(
        "  secret  ",
        domain="example.test",
        save_credentials=False,
    )
    credentials_connector = TianyanConnector.from_credentials(credentials_path="credentials.json")

    assert login_connector.list_devices() == ()
    assert credentials_connector.list_devices() == ()
    assert calls == [
        (
            "login",
            ("secret",),
            {"domain": "example.test", "save_credentials": False},
        ),
        (
            "credentials",
            (),
            {"credentials_path": "credentials.json"},
        ),
    ]


def test_connector_wraps_native_login_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingPlatform:
        @staticmethod
        def login(api_key: str, **kwargs: Any) -> None:
            del kwargs
            raise RuntimeError(f"authentication rejected for {api_key}")

    monkeypatch.setattr(cqlib_tianyan, "TianyanPlatform", FailingPlatform)

    sentinel = "unit-test-secret-that-must-not-leak"
    with pytest.raises(AdapterSubmissionError, match="authentication rejected") as captured:
        TianyanConnector.login(sentinel, save_credentials=False)
    assert sentinel not in str(captured.value)
    assert "<redacted>" in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
