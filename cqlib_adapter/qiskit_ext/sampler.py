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
Sampler module for executing quantum circuits on the TianYan backend.

This module provides a class to sample quantum circuits using the TianYan backend,
leveraging Qiskit's primitives for efficient execution.
"""

from qiskit.primitives import BackendSamplerV2
from qiskit.providers import BackendV2 as Backend, Options

from .tianyan_backend import TianYanBackend


class TianYanSampler(BackendSamplerV2):
    """A sampler class for executing quantum circuits on the TianYan backend.

    This class extends Qiskit's `BackendSamplerV2` to provide sampling functionality
    specifically for the TianYan backend.
    """

    def __init__(self, backend: TianYanBackend, options: dict | None = None) -> None:
        """Initializes the TianYanSampler instance.

         Args:
             backend (TianYanBackend): The TianYan backend to use for sampling.
             options (dict, optional): Additional options for the sampler. Defaults to None.
         """
        super().__init__(backend=backend, options=options)

    @property
    def backend(self) -> Backend:
        """Returns the backend associated with the sampler.

        Returns:
            Backend: The backend used for executing quantum circuits.
        """
        return self._backend
