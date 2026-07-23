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
