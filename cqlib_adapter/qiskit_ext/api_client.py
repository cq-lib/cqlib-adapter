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
import time
from datetime import datetime

import requests
from cqlib.circuits import Circuit
from cqlib.quantum_platform import TianYanPlatform, QuantumLanguage
from cqlib.exceptions import CqlibRequestError


class ApiClient:
    base_url = 'https://qc.zdxlz.com'
    backends = "/qccp-quantum/experiments/quantum/computer/by/user"
    quantum_computer_config = "/qccp-quantum/experiments/quantum/computer/config"
    quantum_machine_config = '/qccp-quantum/sdk/experiment/download/config'
    query_experiment_result = '/qccp-quantum/sdk/experiment/result/find'

    def __init__(self, token: str = None, base_url: str | None = None) -> None:
        if base_url is not None:
            self.base_url = base_url
        if token is None:
            token = os.environ.get("CQLIB_TOKEN", "")
        self.session = requests.Session()
        self._token = token
        if token is not None:
            self._pf = TianYanPlatform(token)
        else:
            self._pf = None

    @property
    def platform(self):
        if self._pf is None:
            self._pf = TianYanPlatform(self._token)
        return self._pf

    def get_backends(self) -> list[dict]:
        response = self.session.get(
            f'{self.base_url}{self.backends}',
            params={'a': str(int(time.time())), 'apiCode': 'byUser', },
            headers={
                'Accept': 'application/json, text/plain, */*',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0',
                'apiCode': 'byUser',
                'requestTime': self._request_time()
            })
        data = response.json()
        if data.get('code') != 200:
            raise CqlibRequestError(f"request error. \n{data.get('message')}")
        return data.get('data', [])

    def get_quantum_computer_config(self, computer_id):
        response = self.session.get(
            f'{self.base_url}{self.quantum_computer_config}',
            params={
                'a': str(int(time.time())),
                'type': 'overview',
                'quantumComputerId': computer_id,
                'label': 'qubits,couplers,coupler_map,disabled_qubits,disabled_couplers'
            },
            headers={
                'Accept': 'application/json, text/plain, */*',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0',
                'apiCode': 'config',
                'requestTime': self._request_time()
            })
        data = response.json()
        return data.get('data', {})

    def get_quantum_machine_config(self, computer_code: str):
        """
        Only quantum machine, not simulator.

        :param computer_code:
        :return:
        """
        return self.platform.download_config(machine=computer_code)

    def submit_job(self, circuits: Circuit | list[Circuit], machine: str, **kwargs):
        if not isinstance(circuits, list):
            circuits = [circuits]

        task_ids = self.platform.submit_experiment(
            [c.as_str() for c in circuits],
            machine_name=machine,
            language=QuantumLanguage.QCIS,
            num_shots=kwargs.get('shots', 1000),
        )
        return task_ids

    def query_job(self, task_ids: list[str]):
        response = self.session.post(
            f'{self.base_url}{self.query_experiment_result}',
            json={"query_ids": task_ids},
            headers={
                'Accept': 'application/json, text/plain, */*',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0',
                'apiCode': 'config',
                'requestTime': self._request_time(),
                "basicToken": self.platform.access_token,
                "Authorization": f'Bearer {self.platform.access_token}'
            })
        data = response.json()
        if data.get('code') != 0:
            raise CqlibRequestError(f"request error {data.get('msg')}")
        return data.get('data').get('experimentResultModelList')

    @staticmethod
    def _request_time():
        return datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p") \
            .replace("/0", "/").lstrip("0").replace(" 0", " ")
