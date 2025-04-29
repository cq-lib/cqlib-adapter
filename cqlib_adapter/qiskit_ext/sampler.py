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

from qiskit.primitives import BackendSamplerV2
from qiskit.providers import BackendV2 as Backend, Options

from .tianyan_backend import TianYanBackend


class TianYanSampler(BackendSamplerV2):

    def __init__(self, backend: TianYanBackend, options: dict | None = None) -> None:
        super().__init__(backend=backend, options=options)

    @property
    def backend(self) -> Backend:
        return self._backend
