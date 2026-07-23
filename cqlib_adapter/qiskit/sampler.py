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

"""Qiskit primitive support built on the Tianyan BackendV2."""

from __future__ import annotations

from typing import Any

from qiskit.primitives import BackendSamplerV2

from .backend import TianyanBackend


class TianyanSampler(BackendSamplerV2):
    """Qiskit BackendSamplerV2 pre-bound to a TianyanBackend."""

    def __init__(
        self,
        backend: TianyanBackend,
        *,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(backend=backend, options=options)


__all__ = ["TianyanSampler"]
