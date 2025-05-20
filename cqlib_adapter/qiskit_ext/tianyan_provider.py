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
TianYan Provider module for managing quantum computing backends.

This module provides a class to interact with the TianYan platform, allowing users to
retrieve and manage quantum computing backends (both quantum computers and simulators).
"""

import os
from pathlib import Path

import dotenv

from .api_client import ApiClient
from .tianyan_backend import TianYanBackend, BackendConfiguration, CqlibAdapterError, \
    BackendStatus, TianYanQuantumBackend, TianYanSimulatorBackend


class TianYanProvider:
    """A provider class for interacting with the TianYan quantum computing platform.

    This class allows users to retrieve and manage backends (quantum computers and simulators)
    available on the TianYan platform.
    """

    def __init__(
            self,
            token: str | None = None,
            *,
            load_dotenv: bool = True,
            dotenv_path: str | Path | None = None,
    ) -> None:
        """Initializes the TianYanProvider instance.

        Args:
            token (str, optional): The authentication token for the TianYan platform.
                If not provided, it will be fetched from the environment variable CQLIB_TOKEN.
            load_dotenv (bool, optional): Whether to load environment variables from a .env file.
                Defaults to True.
            dotenv_path (str | Path, optional): The path to the .env file. Defaults to None.
        """
        if load_dotenv or dotenv_path is not None:
            dotenv.load_dotenv(dotenv_path)

        if token is None:
            token = os.environ.get("CQLIB_TOKEN", "")
        self.token = token

        self.name = "qiskit_adapter"
        self._api_client = ApiClient(token=token)

    def backends(self, simulator: bool = None, online: bool = True, name: str = None):
        """Retrieves a list of backends based on the specified filters.

        Args:
            simulator (bool, optional): Whether to filter for simulator backends.
                If None, no filtering is applied. Defaults to None.
            online (bool, optional): Whether to filter for online backends.
                Defaults to True.
            name (str, optional): The name of the backend to filter for.
                If None, no filtering is applied. Defaults to None.

        Returns:
            list[TianYanBackend]: A list of backends matching the specified filters.
        """
        bs = []
        for data in self._api_client.get_backends():
            cfg = BackendConfiguration.from_api(data, self._api_client)
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
        """Retrieves a specific backend by name.

        Args:
            name (str): The name of the backend to retrieve.

        Returns:
            TianYanBackend: The backend with the specified name.

        Raises:
            CqlibAdapterError: If the backend with the specified name is not found.
        """
        for data in self._api_client.get_backends():
            if data['code'] != name:
                continue
            cfg = BackendConfiguration.from_api(data, self._api_client)
            if cfg.simulator:
                backend = TianYanSimulatorBackend(configuration=cfg, api_client=self._api_client)
            else:
                backend = TianYanQuantumBackend(configuration=cfg, api_client=self._api_client)
            return backend
        raise CqlibAdapterError(f"Backend {name} not found")
