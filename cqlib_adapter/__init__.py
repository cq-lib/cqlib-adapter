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

"""Adapters connecting quantum programming frameworks to cqlib and Tianyan Cloud."""

from ._version import __version__
from .adapters import ADAPTERS, AdapterInfo, available_adapters, get_adapter_info

__all__ = [
    "ADAPTERS",
    "AdapterInfo",
    "__version__",
    "available_adapters",
    "get_adapter_info",
]
