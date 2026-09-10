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

"""Skip TianYan live-backend tests when CQLIB_TOKEN is not available."""
from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.getenv("CQLIB_TOKEN"):
        return

    skip_token = pytest.mark.skip(reason="CQLIB_TOKEN is not set")
    for item in items:
        path = str(item.path).replace("\\", "/")
        if path.endswith("tests/test_qiskit/test_backend.py"):
            item.add_marker(skip_token)
            continue
        params = getattr(getattr(item, "callspec", None), "params", {})
        backend = params.get("backend_name")
        if backend and backend != "default":
            item.add_marker(skip_token)
