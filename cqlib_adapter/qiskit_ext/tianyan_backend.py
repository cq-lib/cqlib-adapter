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

from qiskit.circuit import QuantumCircuit, Parameter, Measure, Barrier
from qiskit.circuit.library.standard_gates import CZGate, RZGate, PhaseGate, HGate, \
    GlobalPhaseGate
from qiskit.providers import BackendV2 as Backend, Options, JobV1, QubitProperties
from qiskit.transpiler import Target, InstructionProperties

from .api_client import ApiClient
from .adapter import to_cqlib
from .gates import X2PGate, X2MGate, Y2MGate, Y2PGate, XY2MGate, XY2PGate
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
        self._target = None
        self._machine_config = {}

    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def max_circuits(self):
        return 50

    @property
    def machine_config(self):
        return self._machine_config

    @property
    def backend_type(self):
        return self.configuration.backend_type

    def run(
            self,
            run_input,
            shots: int = 1024,
            readout_calibration: bool = True,
            **options
    ) -> JobV1:
        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        elif isinstance(run_input, list):
            circuits = run_input
        else:
            raise TypeError(f"Unsupported input type: {type(run_input)}")

        task_ids = self._api_client.submit_job(
            [to_cqlib(circ) for circ in circuits],
            machine=self.configuration.backend_name,
            shots=shots,
        )
        return TianYanJob(
            backend=self,
            job_id=','.join(task_ids),
            api_client=self._api_client,
            shots=shots,
            readout_calibration=readout_calibration,
            **options
        )

    @property
    def target(self):
        return self._target

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}('{self.name}')>"


time_units = {
    's': 1,
    'ms': 1e-3,
    'us': 1e-6,
    'ns': 1e-9
}
frequency_units = {
    'hz': 1,
    'khz': 1e3,
    'mhz': 1e6,
    'ghz': 1e9
}
number_units = {
    '%': 1e-2,
    '': 1
}


class TianYanQuantumBackend(TianYanBackend):

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
        super().__init__(configuration=configuration, api_client=api_client)
        self._machine_config = self._api_client.get_quantum_machine_config(
            self.configuration.backend_name
        )
        target = Target(
            # num_qubits=configuration.n_qubits,
            description=configuration.backend_name,
            qubit_properties=self._make_qubit_properties()
        )
        self._update_cz_gate(target)
        self._update_single_gates(target)
        self._update_measure_gate(target)
        self._update_barrier_gate(target)
        self._target = target

    def _make_qubit_properties(self):
        t1 = self._machine_config['qubit']['relatime']['T1']
        t1_qubits = t1['qubit_used']
        t1_values = t1['param_list']
        t1_unit = time_units.get(t1['unit'].lower())
        t2 = self._machine_config['qubit']['relatime']['T2']
        t2_qubits = t2['qubit_used']
        t2_values = t2['param_list']
        t2_unit = time_units.get(t2['unit'].lower())
        frequency = self._machine_config['qubit']['frequency']['f01']
        frequency_qubits = frequency['qubit_used']
        frequency_values = frequency['param_list']
        frequency_unit = frequency_units.get(frequency['unit'].lower())

        if t1_qubits != t2_qubits != frequency_qubits:
            raise ValueError("t1/t2/frequency qubits are not the same")
        qubit_properties: list[QubitProperties | None] = [
            None for _ in range(self.configuration.n_qubits)
        ]
        for i, q in enumerate(t1_qubits):
            qubit_properties[int(q[1:])] = QubitProperties(
                t1=t1_values[i] * t1_unit,
                t2=t2_values[i] * t2_unit,
                frequency=frequency_values[i] * frequency_unit
            )
        return qubit_properties

    def _update_cz_gate(self, target: Target):
        cz_props = {}
        coupler_map = self._machine_config['overview']['coupler_map']
        gate_errors = self._machine_config['twoQubitGate']['czGate']['gate error']
        error_qubits = gate_errors['qubit_used']
        error_values = gate_errors['param_list']
        error_unit = number_units[gate_errors['unit']]

        for i, q in enumerate(error_qubits):
            q0, q1 = coupler_map[q]
            q0, q1 = int(q0[1:]), int(q1[1:])
            p = InstructionProperties(error=error_values[i] * error_unit, duration=1e-8)
            cz_props[q0, q1] = p
            cz_props[q1, q0] = p
        target.add_instruction(CZGate(), cz_props)

    def _update_single_gates(self, target: Target):
        rz_props = {}
        single_props = {}

        gate_params = self._machine_config['qubit']['singleQubit']['gate error']
        gate_values = gate_params['param_list']
        gate_qubits = gate_params['qubit_used']
        gate_unit = number_units[gate_params['unit']]

        for i, q in enumerate(gate_qubits):
            q_index = (int(q[1:]),)
            rz_props[q_index] = InstructionProperties(error=0, duration=0)
            single_props[q_index] = InstructionProperties(
                error=gate_values[i] * gate_unit,
                duration=0
            )
        target.add_instruction(RZGate(Parameter('theta')), rz_props)
        target.add_instruction(HGate(), single_props.copy())

        target.add_instruction(X2PGate(), single_props.copy())
        target.add_instruction(X2MGate(), single_props.copy())
        target.add_instruction(Y2PGate(), single_props.copy())
        target.add_instruction(Y2MGate(), single_props.copy())
        target.add_instruction(XY2PGate(Parameter('theta')), single_props.copy())
        target.add_instruction(XY2MGate(Parameter('theta')), single_props.copy())

        target.add_instruction(GlobalPhaseGate(Parameter('phase')), {(): None})

    def _update_measure_gate(self, target: Target):
        gate_params = self._machine_config['readout']['readoutArray']['Readout Error']
        gate_values = gate_params['param_list']
        gate_qubits = gate_params['qubit_used']
        gate_unit = number_units[gate_params['unit']]
        props = {
            (int(q[1:]),): InstructionProperties(error=gate_values[i] * gate_unit, duration=0)
            for i, q in enumerate(gate_qubits)
        }
        target.add_instruction(Measure(), props)

    def _update_barrier_gate(self, target: Target):
        # 添加 Barrier 门的逻辑
        target.add_instruction(Barrier, name="barrier")


class TianYanSimulatorBackend(TianYanBackend):

    def __init__(
            self,
            configuration: BackendConfiguration,
            api_client: 'ApiClient',
    ) -> None:
        super().__init__(configuration=configuration, api_client=api_client)
