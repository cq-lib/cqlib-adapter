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
import os
from pathlib import Path

import dotenv

from .tianyan_backend import TianYanBackend, BackendConfiguration, CqlibAdapterError, \
    BackendStatus, TianYanQuantumBackend, TianYanSimulatorBackend
from .api_client import ApiClient


class TianYanProvider:
    def __init__(
            self,
            token: str | None = None,
            *,
            load_dotenv: bool = True,
            dotenv_path: str | Path | None = None,
    ) -> None:
        if load_dotenv or dotenv_path is not None:
            dotenv.load_dotenv(dotenv_path)

        if token is None:
            token = os.environ.get("CQLIB_TOKEN", "")
        self.token = token

        self.name = "qiskit_adapter"
        self._api_client = ApiClient(token=token)

    def backends(self, simulator: bool = None, online: bool = True, name: str = None):
        bs = []
        for data in self._api_client.get_backends():
            cfg = BackendConfiguration.from_api(data)
            if online and cfg.status not in [BackendStatus.running, BackendStatus.calibrating]:
                continue
            if simulator is not None and cfg.simulator != simulator:
                continue
            if name is not None and cfg.backend_name != name:
                continue
            if cfg.simulator:
                backend = TianYanSimulatorBackend(
                    configuration=cfg,
                    api_client=self._api_client,
                )
            else:
                backend = TianYanQuantumBackend(
                    configuration=cfg,
                    api_client=self._api_client,
                )
            bs.append(backend)
        return bs

    def backend(self, name: str) -> TianYanBackend:
        for b in self.backends():
            if b.name == name:
                return b
        raise CqlibAdapterError(f"Backend {name} not found")
