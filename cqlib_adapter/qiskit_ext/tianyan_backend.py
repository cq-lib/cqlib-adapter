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

import json
from collections import namedtuple
from datetime import datetime
from enum import Enum, IntEnum
from typing import Literal, TypeAlias

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2 as Backend, Options, Job, JobV1
from qiskit.transpiler import Target

from .api_client import ApiClient
from .adapter import to_cqlib
from .job import TianYanJob


class BackendType(IntEnum):
    quantum_computer = 0
    simulator = 1
    offline_simulator = 2


class BackendStatus(IntEnum):
    running = 0
    calibrating = 1
    under_maintenance = 2
    offline = 3


class CqlibAdapterError(Exception):
    """Base class for exceptions in this module."""
    pass


GateConfig = namedtuple('GateConfig', ['name', 'params', 'coupling_map'])

gate_parameters = {
    'rx': 1,
    'ry': 1,
    'rz': 1,
    'rxy': 2,
    'xy2p': 1,
    'xy2m': 1,
}


class BackendConfiguration:
    def __init__(
            self,
            backend_id: str,
            backend_name: str,
            n_qubits: int,
            basis_gates: list,
            gates: list,
            local: bool,
            simulator: bool,
            conditional: bool,
            coupling_map: list,
            status: BackendStatus = None,
            supported_instructions: list[str] = None,
            credits_required: bool = None,
            online_date: datetime = None,
            display_name: str = None,
            description: str = None,
            tags: list = None,
            backend_type=None,
            machine_config=None,
            **kwargs,
    ):
        self.backend_id = backend_id
        self.backend_name = backend_name
        self.n_qubits = n_qubits
        self.basis_gates = basis_gates
        self.gates = gates
        self.simulator = simulator
        self.conditional = conditional
        self.coupling_map = coupling_map
        self.local = local
        self.status = status
        self.supported_instructions = supported_instructions
        self.credits_required = credits_required
        self.online_date = online_date
        self.display_name = display_name
        self.description = description
        self.tags = tags
        self.properties = kwargs
        self.machine_config = machine_config
        self.backend_type = backend_type

    @staticmethod
    def from_dict(data: dict[str: str | int | list]) -> "BackendConfiguration":
        return BackendConfiguration(**data)

    @staticmethod
    def from_api(data: dict[str: str | int | list]) -> "BackendConfiguration":
        disabled_qubits = [q for q in data['disabledQubits'].split(',') if q]
        disabled_couplers = [g for g in data['disabledCouplers'].split(',') if g]
        backend_id = data['id']
        n_qubits = data['bitWidth']

        if data['labels'] == '1':
            backend_type = BackendType.quantum_computer
        else:
            backend_type = BackendType.simulator

        if backend_type == BackendType.quantum_computer:
            qpu = ApiClient().get_quantum_computer_config(backend_id)
            qubits = [int(q[1:]) for q in qpu['qubits'] if q not in disabled_qubits]
            coupling_map = []

            for k, q in qpu['coupler_map'].items():
                if k in disabled_couplers:
                    continue
                q0, q1 = q
                if q0 in disabled_qubits or q1 in disabled_qubits:
                    continue
                coupling_map.append([int(q0[1:]), int(q1[1:])])
        else:
            qubits = [i for i in range(n_qubits)]
            # todo: 全振幅仿真机，比特太多了。先限制一下
            coupling_map = [[i, j] for i in range(min(n_qubits, 100)) for j in range(i)]
        basis_gates = []
        gates = []

        for gate in data['baseGate']:
            name = gate['qcis'].lower()
            rule = gate['rule']
            # to qiskit gate name
            if name == 'i':
                name = 'id'
            elif name == 'b':
                name = 'barrier'
            elif name == 'm':
                name = 'measure'
            basis_gates.append(name)

            try:
                rule = json.loads(rule)
            except json.JSONDecodeError:
                pass

            # coupler gate
            if isinstance(rule, dict) and 'topology' in rule:
                gate_coupling_map = coupling_map
            # single gate
            else:
                gate_coupling_map = [[q] for q in qubits]

            gates.append(GateConfig(
                name=name,
                params=[f'p_{i}' for i in range(gate_parameters.get(name, 0))],
                coupling_map=gate_coupling_map
            ))

        data = {
            'backend_id': backend_id,
            'backend_name': data['code'],
            'n_qubits': data['bitWidth'],
            'basis_gates': basis_gates,
            'gates': gates,
            'local': False,
            'simulator': backend_type in [BackendType.simulator, BackendType.offline_simulator],
            'supported_instructions': basis_gates,
            'credits_required': data['isToll'] == 2,
            'status': BackendStatus(data['status']),
            'online_date': datetime.strptime(data['createTime'], '%Y-%m-%d %H:%M:%S'),
            'display_name': data['name'],
            'description': '',
            'conditional': False,
            'coupling_map': coupling_map,
            'backend_type': backend_type,
        }
        return BackendConfiguration.from_dict(data)


class TianYanBackend(Backend):

    def __init__(
            self,
            configuration: BackendConfiguration,
            api_client: 'ApiClient',
    ) -> None:
        """
        Initialize the backend.

        Args:
            configuration:
            api_client:
        """
        super().__init__(name=configuration.backend_name, )
        self.configuration = configuration
        self.resource_id = configuration.backend_id
        self.resource_type = configuration.backend_id
        self.simulator = configuration.simulator
        self._api_client = api_client
        self._target = Target()
        self._machine_config = None

    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def max_circuits(self):
        return 50

    @property
    def machine_config(self):
        if self.backend_type == BackendType.simulator:
            return {}
        if self._machine_config is None:
            self._machine_config = self._api_client.get_quantum_machine_config(self.configuration.backend_name)
        return self._machine_config

    @property
    def backend_type(self):
        return self.configuration.backend_type

    def run(self, run_input, **options) -> JobV1:
        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        elif isinstance(run_input, list):
            circuits = run_input
        else:
            raise TypeError(f"Unsupported input type: {type(run_input)}")

        cqlib_circuits = [to_cqlib(circ) for circ in circuits]
        if self.simulator:
            cs = [c.as_str() for c in cqlib_circuits]
        else:
            cs = [c.qcis for c in cqlib_circuits]

        task_ids = self._api_client.submit_job(
            cs,
            machine=self.configuration.backend_name,
            **options
        )
        return TianYanJob(
            backend=self,
            job_id=','.join(task_ids),
            api_client=self._api_client,
            **options
        )

    @property
    def target(self):
        return self._target

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}('{self.name}')>"
