# This code is part of cqlib.
#
# Copyright (C) 2025 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Qiskit extension module for TianYan quantum platform integration.
"""

from .adapter import to_cqlib
from .api_client import ApiClient
from .gates import X2PGate, X2MGate, Y2PGate, Y2MGate, XY2MGate, XY2PGate, RxyGate
from .job import TianYanJob
from .sampler import TianYanSampler
from .tianyan_backend import TianYanBackend
from .tianyan_provider import TianYanProvider

__all__ = [
    "to_cqlib",
    "ApiClient",

    'X2PGate',
    'X2MGate',
    'Y2PGate',
    'Y2MGate',
    'XY2PGate',
    'XY2MGate',
    'RxyGate',

    "TianYanJob",
    "TianYanSampler",
    "TianYanProvider",
    "TianYanBackend",
]
