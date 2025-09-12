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

"""PennyLane device implementation for CQLib quantum computing backend."""

import json
import os
from typing import Dict, List, Union

import cqlib
import numpy as np
import pennylane as qml
from cqlib import TianYanPlatform
from cqlib.mapping import transpile_qcis
from cqlib.simulator import StatevectorSimulator
from cqlib.utils import qasm2
from pennylane.devices import Device
from pennylane.tape import QuantumScript, QuantumScriptOrBatch


class CQLibDevice(Device):
    """Custom quantum device class implementing PennyLane device interface using CQLib backend.

    This device provides an interface between PennyLane and various CQLib backends,
    including local simulators, TianYan cloud simulators, and TianYan hardware.
    """

    # Device metadata
    short_name = "cqlib.device"
    config_filepath = os.path.join(os.path.dirname(__file__), "cqlib_config.toml")

    def __init__(self, wires, shots=None, cqlib_backend_name="default", login_key=None):
        """Initializes the CQLib device.

        Args:
            wires: Number of quantum wires (qubits) for the device.
            shots: Number of measurement shots. If None, uses analytic computation.
            cqlib_backend_name: Name of the CQLib backend to use. Defaults to "default".
            login_key: TianYan platform login key (if required for cloud backends).
        """
        super().__init__(wires=wires, shots=shots)
        self.num_wires = wires
        self.num_shots = shots
        self.machine_name = cqlib_backend_name

        # Initialize TianYan platform connection for non-default backends
        if cqlib_backend_name != "default":
            self.cqlib_backend = TianYanPlatform(
                login_key=login_key, machine_name=cqlib_backend_name
            )

    @property  
    def name(self):
        """Returns the device name.

        Returns:
            str: The name of the device.
        """
        return "CQLib Quantum Device"

    @classmethod
    def capabilities(cls):
        """Returns the device capabilities configuration.

        Returns:
            dict: Dictionary containing supported features and capabilities.
        """
        capabilities = super().capabilities().copy()

        capabilities.update(
            model="qubit",
            supports_inverse_operations=False,
            supports_analytic_computation=True,
            supports_finite_shots=True,
            returns_state=False,
            passthru_devices={
                "autograd": "default.qubit.autograd",
                "tf": "default.qubit.tf",
                "torch": "default.qubit.torch",
                "jax": "default.qubit.jax",
            },
        )

        return capabilities

    def execute(self, circuits: QuantumScriptOrBatch, execution_config=None):
        """Executes quantum circuits on the target backend."""
        if isinstance(circuits, qml.tape.QuantumScript):
            circuits = [circuits]
        
        results = []

        for circuit in circuits:
            wires = circuit.wires.labels
            # print(circuit)
            has_state_measurement = any(
            isinstance(m, qml.measurements.StateMP) 
            for m in circuit.measurements
        )

            if has_state_measurement and self.shots:
                raise ValueError(
                    "State measurement requires shots=None (analytic mode). "
                    f"Current shots: {self.shots}"
                )
 
            new_ops = list(circuit.operations)
            for measurement in circuit.measurements:
                if isinstance(measurement, qml.measurements.ExpectationMP):
                    obs = measurement.obs
                    
                    if hasattr(obs, 'name') and obs.name == "Prod":
                        for op in obs.operands:
                            if op.name == "PauliX":
                                new_ops.append(qml.Hadamard(wires=op.wires))
                            elif op.name == "PauliY":
                                new_ops.append(qml.adjoint(qml.S)(wires=op.wires))
                                new_ops.append(qml.Hadamard(wires=op.wires))
                    # 处理单泡利算子
                    elif hasattr(obs, 'name'):
                        if obs.name == "PauliX":
                            new_ops.append(qml.Hadamard(wires=obs.wires))
                        elif obs.name == "PauliY":
                            new_ops.append(qml.adjoint(qml.S)(wires=obs.wires))
                            new_ops.append(qml.Hadamard(wires=obs.wires))

            # Convert circuit to QCIS format
            qasm_str = circuit.to_openqasm()
            cqlib_circuit = qasm2.loads(qasm_str)
            cqlib_qcis = cqlib_circuit.qcis
            circuit = qml.tape.QuantumScript(
                new_ops, circuit.measurements, shots=circuit.shots
            )

            # Execute based on backend type
            if self._is_tianyan_hardware():
                compiled_circuit = transpile_qcis(cqlib_qcis, self.cqlib_backend)
                query_id = self.cqlib_backend.submit_experiment(
                    compiled_circuit[0].qcis, num_shots=self.num_shots
                )
                raw_result = self.cqlib_backend.query_experiment(
                    query_id, readout_calibration=True
                )
                result = extract_probability(raw_result, num_wires=self.num_wires)

            elif self._is_tianyan_simulator():
                query_id = self.cqlib_backend.submit_experiment(
                    cqlib_qcis, num_shots=self.num_shots
                )
                raw_result = self.cqlib_backend.query_experiment(query_id)
                result = extract_probability(raw_result, num_wires=self.num_wires)

            else:
                calc_type = circuit.measurements
                result = self._execute_on_simulator(cqlib_circuit,calc_type,wires)

            # Process measurement results
            circuit_results = self._process_measurements(circuit, result)
            results.append(circuit_results)

        return results

    def _is_tianyan_hardware(self):
        """Checks if the current backend is TianYan hardware.

        Returns:
            bool: True if the backend is TianYan hardware, False otherwise.
        """
        return self.machine_name in {
            "tianyan24",
            "tianyan504",
            "tianyan176-2",
            "tianyan176",
        }

    def _is_tianyan_simulator(self):
        """Checks if the current backend is TianYan simulator.

        Returns:
            bool: True if the backend is TianYan simulator, False otherwise.
        """
        return self.machine_name in {
            "tianyan_sw",
            "tianyan_s",
            "tianyan_tn",
            "tianyan_tnn",
            "tianyan_sa",
            "tianyan_swn",
        }

    def _execute_on_simulator(self, circuit, calc_type, wires):
        """Executes the circuit on a local simulator.
        
        Args:
            circuit: Circuit to execute.
            calc_type: Measurement type.
            wires: Quantum wire ordering for result rearrangement.
        
        Returns:
            Union[dict, np.ndarray]: Sampling results or state vector.
        """
        if len(calc_type) > 1:
            raise ValueError("Cannot use measurement more than once.")

        simulator = StatevectorSimulator(circuit)
        
        if isinstance(calc_type[0], qml.measurements.StateMP):
            
            # 获取原始状态向量字典
            original_statevector = simulator.statevector()
            
            # 获取量子比特数
            num_qubits = len(next(iter(original_statevector.keys())))
            
            # 生成所有可能的基态字符串（小端序）
            all_basis_states = [format(i, '0' + str(num_qubits) + 'b')[::-1] 
                            for i in range(2**num_qubits)]
            
            # 根据原始顺序提取振幅
            statevector = [original_statevector.get(basis_state, 0+0j) 
                        for basis_state in all_basis_states]
            
            # 如果提供了wires参数且需要重排，则进行重排
            if wires and len(wires) == num_qubits:
                statevector = self._rearrange_statevector(statevector, wires, num_qubits)
            
            # 转换为 NumPy 数组
            statevector_np = np.array(statevector)
            return statevector_np
        
        elif isinstance(calc_type[0], qml.measurements.ProbabilityMP):
            probs = simulator.probs()
            if wires is not None and len(wires) == int(np.log2(len(probs))):
                probs = self._rearrange_probabilities(probs, wires)
            return probs
        
        elif isinstance(calc_type[0], qml.measurements.ExpectationMP):
            return simulator.sample()
        
        else:
            raise TypeError("Unknown Error!")

    def _rearrange_statevector(self, statevector, wires, num_qubits):
        """重排状态向量以匹配指定的量子比特顺序（大端序）
        
        Args:
            statevector: 原始状态向量（numpy数组）
            wires: 新顺序，例如 (2,0,1) 表示：
                - 新位置0对应原始位置2的量子比特
                - 新位置1对应原始位置0的量子比特  
                - 新位置2对应原始位置1的量子比特
            num_qubits: 量子比特数量
        """
        rearranged_statevector = np.zeros_like(statevector, dtype=complex)
        
        for new_index in range(len(statevector)):
            # 将新索引转换为二进制字符串（大端序：高位在前，不需要反转）
            new_basis = format(new_index, '0' + str(num_qubits) + 'b')
            
            # 根据wires映射找到对应的原始基态
            original_basis = ['0'] * num_qubits
            for new_pos in range(num_qubits):
                original_pos = wires[new_pos]  # 新位置new_pos对应原始位置original_pos
                original_basis[original_pos] = new_basis[new_pos]
            
            # 转换回原始索引（大端序：直接转换）
            original_basis_str = ''.join(original_basis)
            original_index = int(original_basis_str, 2)
            
            # 将原始状态的振幅放到新位置
            rearranged_statevector[new_index] = statevector[original_index]
        
        return rearranged_statevector


    def _rearrange_probabilities(self, probs_dict, wires):
        """
        重排概率分布字典以匹配指定的量子比特顺序（大端序）
        
        参数:
            probs_dict: dict[str, float]
                原始概率分布字典，key 是二进制字符串（如 '010'），
                最高位在左边，最低位在右边。
            wires: list[int]
                新的量子比特顺序映射，例如 [2,1,0] 表示把原 qubit2 放到新位置0，
                qubit1 放到新位置1，qubit0 放到新位置2。
        返回:
            dict[str, float]
                重排后的概率分布字典。
        """
        num_qubits = len(next(iter(probs_dict.keys())))
        rearranged_probs = {}

        for original_basis, prob in probs_dict.items():
            if prob == 0.0:
                continue

            new_basis = ['0'] * num_qubits
            for new_pos in range(num_qubits):
                original_pos = wires[new_pos]           # 原始 qubit 索引
                str_index = num_qubits - 1 - original_pos  # 映射到字符串下标
                new_basis[new_pos] = original_basis[str_index]

            new_basis_str = ''.join(new_basis)
            rearranged_probs[new_basis_str] = prob

        return rearranged_probs

    def _process_measurements(
        self, circuit: QuantumScript, raw_result: Union[dict, List[dict]]
    ) -> Union[float, np.ndarray, List[Union[float, np.ndarray]]]:
        """Processes measurement results based on circuit measurement operations.

        Args:
            circuit: PennyLane quantum circuit.
            raw_result: Raw result from backend (simulator or hardware).

        Returns:
            Union[float, np.ndarray, List]: Measurement results (probabilities or
            expectation values).

        Raises:
            NotImplementedError: If an unsupported measurement type is encountered.
        """
        results = []
        for measurement in circuit.measurements:
            if isinstance(measurement, qml.measurements.ExpectationMP):
                results.append(self._process_expectation(measurement, raw_result))
            elif isinstance(measurement, qml.measurements.ProbabilityMP):
                results.append(self._process_probability(measurement, raw_result))
            elif isinstance(measurement, qml.measurements.StateMP):
                results.append(self._process_state(measurement, raw_result))
            
            else:
                raise NotImplementedError(
                    f"Measurement type {type(measurement).__name__} is not supported"
                )

        # Return single result directly if only one measurement
        return results[0] if len(results) == 1 else results

    def _process_expectation(self, measurement, raw_result) -> float:
        """Processes expectation value measurements for single and multi-Pauli observables.
        
        Args:
            measurement: Expectation measurement operation.
            raw_result: Raw result data.
        
        Returns:
            float: Processed expectation value.
        """
        obs = measurement.obs
        
        # 处理多泡利张量积的情况
        if hasattr(obs, 'name') and obs.name == "Prod":
            # 提取所有泡利算子
            pauli_operators = []
            if hasattr(obs, 'operands'):
                pauli_operators = obs.operands
            elif hasattr(obs, '_obs'):
                pauli_operators = obs._obs
            
            # 检查是否都是泡利Z算子（当前实现假设）
            for op in pauli_operators:
                if not hasattr(op, 'name') or not op.name.startswith('Pauli'):
                    raise NotImplementedError(
                        f"Multi-observable expectation only supports Pauli operators, got {op.name}"
                    )
            
            return self._process_multi_pauli_expectation(pauli_operators, raw_result)
        
        # 处理单泡利算子的情况（保持原有逻辑）
        elif hasattr(obs, 'name') and obs.name in ["PauliZ", "PauliX", "PauliY"]:
            if obs.name != "PauliZ":
                raise NotImplementedError(
                    f"Single observable expectation for {obs.name} is not yet supported"
                )
            
            if isinstance(raw_result, list):
                return self.process_results(raw_result)
            elif isinstance(raw_result, dict):
                local_expectation = self.process_results_local(raw_result)
                return local_expectation[measurement.wires[0]]
            else:
                raise ValueError(f"Unsupported raw_result type: {type(raw_result)}")
        else:
            raise NotImplementedError(
                f"Expectation for {type(obs).__name__} is not supported"
            )

    def _process_multi_pauli_expectation(self, pauli_operators, raw_result) -> float:
        """Processes expectation value for tensor products of Pauli operators.
        
        Args:
            pauli_operators: List of Pauli operators in the tensor product.
            raw_result: Raw result data from backend.
        
        Returns:
            float: Joint expectation value.
        
        Raises:
            ValueError: If raw_result format is invalid.
        """
        if isinstance(raw_result, dict):
            # 处理本地模拟器的结果
            total_shots = sum(raw_result.values())
            expectation = 0.0
            
            for bitstring, count in raw_result.items():
                # 计算这个比特串对应的特征值乘积
                eigenvalue_product = 1.0
                
                for op in pauli_operators:
                    qubit_idx = op.wires[0]  # 假设每个泡利算子作用在单个量子比特上
                    bit_value = int(bitstring[-qubit_idx - 1])  # 获取对应量子比特的测量结果
                    
                    # 泡利Z的特征值：|0⟩ → +1, |1⟩ → -1
                    eigenvalue = 1.0 if bit_value == 0 else -1.0
                    eigenvalue_product *= eigenvalue
                
                expectation += (count / total_shots) * eigenvalue_product
            
            return expectation
        
        elif isinstance(raw_result, list):
            # 处理云端模拟器/硬件的结果（概率分布形式）
            try:
                if isinstance(raw_result[0], dict) and 'probability' in raw_result[0]:
                    probability_dict = raw_result[0]['probability']
                    if isinstance(probability_dict, str):
                        probability_dict = json.loads(probability_dict)
                    
                    expectation = 0.0
                    for state, probability in probability_dict.items():
                        eigenvalue_product = 1.0
                        
                        for op in pauli_operators:
                            qubit_idx = op.wires[0]
                            bit_value = int(state[-qubit_idx - 1])  # 注意比特顺序
                            
                            eigenvalue = 1.0 if bit_value == 0 else -1.0
                            eigenvalue_product *= eigenvalue
                        
                        expectation += probability * eigenvalue_product
                    
                    return expectation
                else:
                    raise ValueError("Invalid probability format in raw_result")
                    
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid raw_result format: {error}") from error
        
        else:
            raise ValueError(f"Unsupported raw_result type: {type(raw_result)}")

    def _process_probability(self, measurement, raw_result) -> np.ndarray:
        """Processes probability measurements.

        Args:
            measurement: Probability measurement operation.
            raw_result: Raw result data.

        Returns:
            np.ndarray: Probability distribution array.

        Raises:
            ValueError: If raw_result format is invalid or probabilities don't sum to 1.
        """
        num_wires = len(measurement.wires)
        probabilities = np.zeros(2**num_wires)

        if isinstance(raw_result, dict):
            total_shots = sum(raw_result.values())
            for bitstring, count in raw_result.items():
                index = int(bitstring[::-1], 2)
                probabilities[index] = count / total_shots

        elif isinstance(raw_result, list):
            try:
                probability_dict = json.loads(raw_result[0]["probability"])
                for bitstring, probability in probability_dict.items():
                    index = int(bitstring[::-1], 2)
                    probabilities[index] = probability
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid raw_result format: {error}") from error
        else:
            raise ValueError(f"Unsupported raw_result type: {type(raw_result)}")

        # Verify probabilities sum to 1 (with tolerance for numerical errors)
        if not np.isclose(np.sum(probabilities), 1.0, rtol=1e-5):
            raise ValueError(f"Probabilities do not sum to 1: {np.sum(probabilities)}")

        return probabilities

    def _process_state(self, measurement, raw_result) -> np.ndarray:
        """Processes state vector measurements.
        
        Args:
            measurement: State measurement operation.
            raw_result: Raw result data from backend (should be the state vector).
        
        Returns:
            np.ndarray: State vector array.
        
        Raises:
            ValueError: If state measurement is requested with finite shots.
            NotImplementedError: If backend doesn't support state vector simulation.
        """
        # 检查是否使用有限shots
        if self.shots:
            raise ValueError(
                "State vector measurement is only supported with shots=None (analytic mode). "
                f"Current shots setting: {self.shots}"
            )
        
        if self.machine_name not in ['default','tianyan_sw','tianyan_tn']:
            raise ValueError(
                "The backend you have chosen does not support state vector computation. "
                f"Current backend: {self.machine_name}"
            )


        if isinstance(raw_result, np.ndarray):
            return raw_result
        else:
            raise ValueError(
                f"Expected state vector but got {type(raw_result)}. "
                "Check _execute_on_simulator implementation."
            )

    def process_results(self, raw_result):
        """Processes expectation value results from hardware or cloud simulator.

        Args:
            raw_result: Raw result data.

        Returns:
            float: PauliZ expectation value.
        """
        probability_dict = json.loads(raw_result[0]["probability"])

        expectation = 0.0
        for state, probability in probability_dict.items():
            if state[0] == "0":  # |0⟩ state corresponds to Z eigenvalue +1
                expectation += probability
            else:  # |1⟩ state corresponds to Z eigenvalue -1
                expectation -= probability

        return expectation

    def process_results_local(self, raw_result):
        """Processes expectation value results from local simulator.

        Args:
            raw_result: Raw result data.

        Returns:
            dict: PauliZ expectation values for each qubit.
        """
        total_shots = sum(raw_result.values())
        num_qubits = len(next(iter(raw_result.keys())))
        z_expectations = {}

        for qubit in range(num_qubits):
            count_0, count_1 = 0, 0
            for bitstring, count in raw_result.items():
                bit = int(bitstring[-qubit - 1])
                if bit == 0:
                    count_0 += count
                else:
                    count_1 += count

            z_expectations[qubit] = (count_0 - count_1) / total_shots

        return z_expectations

    def preprocess_transforms(self, execution_config=None):
        """Defines the preprocessing transformation pipeline.

        Returns:
            TransformProgram: Preprocessing transformation program.
        """
        program = qml.transforms.core.TransformProgram()
        program.add_transform(
            qml.devices.preprocess.validate_device_wires,
            wires=self.wires,
            name=self.short_name,
        )
        program.add_transform(
            qml.devices.preprocess.validate_measurements, name=self.short_name
        )
        program.add_transform(
            qml.devices.preprocess.decompose,
            stopping_condition=self.supports_operation,
            name=self.short_name,
        )
        return program

    def supports_operation(self, operation):
        """Checks if a specific quantum operation is supported.

        Args:
            operation: Quantum operation to check.

        Returns:
            bool: True if the operation is supported, False otherwise.
        """
        supported_operations = {
            "PauliX",
            "PauliY",
            "PauliZ",
            "Hadamard",
            "S",
            "T",
            "RX",
            "RY",
            "RZ",
            "CNOT",
            "CZ",
        }
        return getattr(operation, "name", None) in supported_operations

    def __repr__(self):
        """Returns string representation of the device.

        Returns:
            str: String representation of the device.
        """
        return f"<{self.name} device (wires={self.num_wires}, shots={self.shots})>"


def extract_probability(
    json_data: List[Dict[str, Union[Dict[str, float], list]]], num_wires: int
) -> Dict:
    """Extracts probability distribution from JSON data.

    Args:
        json_data: JSON data containing measurement results, expected to be a list
                   containing dictionaries with 'probability' field.
        num_wires: Number of quantum wires (qubits) in the circuit*(Reserved for future update).

    Returns:
        Dict: Probability distribution for each quantum state.

    Raises:
        ValueError: If JSON data is invalid or missing probability field.
    """
    if not isinstance(json_data, list) or not json_data:
        raise ValueError("json_data must be a non-empty list")
    if not isinstance(json_data[0], dict):
        raise ValueError("json_data[0] must be a dictionary")
    if "probability" not in json_data[0]:
        raise ValueError("probability field missing in json_data[0]")

    try:
        probability_dict = json_data[0]["probability"]
        if isinstance(probability_dict, str):
            probability_dict = json.loads(probability_dict)
        if not isinstance(probability_dict, dict):
            raise ValueError("probability field must be a dictionary")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid probability field format: {error}") from error

    return probability_dict
