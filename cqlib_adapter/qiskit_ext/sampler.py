from qiskit.primitives import BackendSamplerV2
from qiskit.providers import BackendV2 as Backend, Options

from .tianyan_backend import TianYanBackend


class TianYanSampler(BackendSamplerV2):

    def __init__(self, backend: TianYanBackend, options: dict | None = None) -> None:
        super().__init__(backend=backend, options=options)

    @property
    def backend(self) -> Backend:
        return self._backend
