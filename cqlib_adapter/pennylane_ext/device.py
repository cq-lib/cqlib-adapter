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

from pennylane.ops.op_math import Sum, Prod, SProd
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
            supports_analytic_computation=False,
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
        """Executes quantum circuits on the target backend.

        Args:
            circuits: Quantum circuit(s) to execute (single or batch).
            execution_config: Execution configuration parameters.

        Returns:
            list: List of measurement results for each circuit.
        """
        # Ensure circuits is a list (even if single circuit is provided)
        if isinstance(circuits, qml.tape.QuantumScript):
            circuits = [circuits]

        results = []

        for circuit in circuits:
            # Insert basis change gates for PauliX/PauliY measurements
            new_ops = list(circuit.operations)

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
                result = self._execute_on_simulator(cqlib_circuit)

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

    def _execute_on_simulator(self, circuit):
        """Executes the circuit on a local simulator.

        Args:
            circuit: Circuit to execute.

        Returns:
            dict: Sampling results from the simulator.
        """
        simulator = StatevectorSimulator(circuit)
        return simulator.sample()

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
                results.append(self._process_expectation(measurement, raw_result, circuit))
            elif isinstance(measurement, qml.measurements.ProbabilityMP):
                results.append(self._process_probability(measurement, raw_result))
            else:
                raise NotImplementedError(
                    f"Measurement type {type(measurement).__name__} is not supported"
                )

        # Return single result directly if only one measurement
        return results[0] if len(results) == 1 else results

    def _process_expectation(self, measurement, raw_result, circuit=None) -> float:
        """Processes expectation value measurements."""
        obs = measurement.obs

        # 统一把 circuit 传下去（用于需要重跑的情况）
        return self._expval_recursive(obs, raw_result, circuit)

    def _expval_recursive(self, obs, raw_result, circuit):
        # Hamiltonian（旧版常见）
        if isinstance(obs, qml.Hamiltonian):
            total = 0.0
            for coeff, term in zip(obs.coeffs, obs.ops):
                total += coeff * self._expval_recursive(term, raw_result, circuit)
            return total

        # 和：Sum(op1, op2, ...)
        if isinstance(obs, Sum):
            return sum(self._expval_recursive(term, raw_result, circuit) for term in obs.operands)

        # 张量积：Prod(op1, op2, ...)
        if isinstance(obs, Prod):
            # 注意：纠缠态下 ⟨A⊗B⟩ ≠ ⟨A⟩⟨B⟩
            # 我们要把整个张量算符当成一个“Pauli 字符串”整体来算
            return self._expval_pauli_tensor(obs.operands, raw_result, circuit)

        # 标量乘：SProd(c, op)
        if isinstance(obs, SProd):
            return float(obs.scalar) * self._expval_recursive(obs.base, raw_result, circuit)

        # 单一可观测量（PauliX/Y/Z/Identity）
        return self._process_single_observable(obs, raw_result, circuit)
    

    def _process_single_observable(self, obs, raw_result, circuit=None) -> float:
        name = getattr(obs, "name", None)

        # 恒等算符：返回 1.0
        if name in ("Identity", "IdentityMP"):
            return 1.0

        # 单比特 Pauli：Z/X/Y
        if name in ("PauliZ", "PauliX", "PauliY"):
            # 单比特时，我们可以把它当作长度为1的“Pauli 字符串”统一处理
            return self._expval_pauli_tensor([obs], raw_result, circuit)

        # 其它算符（如 Projector、Hermitian 等）当前不支持
        raise NotImplementedError(f"Expectation for {name} is not supported")
    

    def _expval_pauli_tensor(self, operands, raw_result, circuit):
        """计算一个 Pauli 张量（可能跨多比特，可能包含 X/Y/Z）的期望值。"""

        # 收集每个算符的 (axis, wire)
        axes = []  # list of ('X'/'Y'/'Z', int_wire)
        for term in operands:
            n = getattr(term, "name", None)
            if n == "PauliX":
                axes.append(("X", term.wires[0]))
            elif n == "PauliY":
                axes.append(("Y", term.wires[0]))
            elif n == "PauliZ":
                axes.append(("Z", term.wires[0]))
            elif n in ("Identity", "IdentityMP"):
                # 对应位恒等不影响乘积
                continue
            else:
                raise NotImplementedError(f"Unsupported operator in tensor: {n}")

        if not axes:
            return 1.0

        # 情况 A：全是 Z —— 直接用已有 raw_result（一次执行的计数/概率）
        if all(ax == "Z" for ax, _ in axes):
            return self._expval_zz_string_from_raw(raw_result, [w for _, w in axes])

        # 情况 B：包含 X/Y —— 需要按该项做基变换重跑一次
        if circuit is None:
            # 为了健壮：如果未传入 circuit，我们无法重跑，只能报错
            raise NotImplementedError("X/Y terms require circuit to re-execute with basis change")

        return self._expval_via_basis_rotation(circuit, axes)
    
    def _expval_zz_string_from_raw(self, raw_result, wires):
        """从 raw_result（dict 或 list 包含 probability）计算 Z⊗Z⊗... 的期望值。"""
        if isinstance(raw_result, dict):
            total = sum(raw_result.values())
            if total == 0:
                return 0.0
            acc = 0.0
            for bitstring, cnt in raw_result.items():
                # 注意你的位序：你此前用的是 bitstring[::-1]
                eigen = 1
                for w in wires:
                    bit = int(bitstring[-w-1])  # 低位是 wire 0
                    eigen *= (1 if bit == 0 else -1)
                acc += eigen * cnt
            return acc / total

        elif isinstance(raw_result, list):
            # 硬件/云返回 probability JSON（字符串或 dict）
            prob_dict = raw_result[0]["probability"]
            if isinstance(prob_dict, str):
                prob_dict = json.loads(prob_dict)

            acc = 0.0
            for bitstring, p in prob_dict.items():
                eigen = 1
                for w in wires:
                    bit = int(bitstring[-w-1])
                    eigen *= (1 if bit == 0 else -1)
                acc += eigen * p
            return acc

        else:
            raise ValueError(f"Unsupported raw_result type: {type(raw_result)}")
        
    
    def _expval_via_basis_rotation(self, circuit: QuantumScript, axes):
        """对包含 X/Y 的 Pauli 字符串，复制电路、加入对应基变换并执行一次，再用 Z 基统计计算期望。"""

        # 1) 复制原有操作
        new_ops = list(circuit.operations)

        # 2) 在对应 wire 上加入基变换
        for ax, w in axes:
            if ax == "X":
                new_ops.append(qml.Hadamard(wires=w))
            elif ax == "Y":
                new_ops.append(qml.adjoint(qml.S)(wires=w))
                new_ops.append(qml.Hadamard(wires=w))
            # Z 不需要加门

        # 3) 构造测量：在所有涉及的 wires 上做 Probability（拿到全局分布最安全）
        #    也可以只测所有 qubits 的概率，保持简单：我们直接测全体概率
        new_measurements = [qml.probs(wires=range(self.num_wires))]

        # 4) 组装新 tape 并执行（重用你原有的 QASM 转换/后端路径）
        rotated_circuit = qml.tape.QuantumScript(new_ops, new_measurements, shots=circuit.shots)

        # —— 以下与 execute() 里一致：转换为 QCIS 并执行
        qasm_str = rotated_circuit.to_openqasm()
        cqlib_circuit = qasm2.loads(qasm_str)
        cqlib_qcis = cqlib_circuit.qcis

        if self._is_tianyan_hardware():
            compiled_circuit = transpile_qcis(cqlib_qcis, self.cqlib_backend)
            query_id = self.cqlib_backend.submit_experiment(compiled_circuit[0].qcis, num_shots=self.num_shots)
            raw = self.cqlib_backend.query_experiment(query_id, readout_calibration=True)
            prob = extract_probability(raw, num_wires=self.num_wires)
        elif self._is_tianyan_simulator():
            query_id = self.cqlib_backend.submit_experiment(cqlib_qcis, num_shots=self.num_shots)
            raw = self.cqlib_backend.query_experiment(query_id)
            prob = extract_probability(raw, num_wires=self.num_wires)
        else:
            sim = StatevectorSimulator(cqlib_circuit)
            raw = sim.sample()  # dict: bitstring -> count
            # 本地模拟器时直接从计数算
            return self._expval_zz_string_from_raw(raw, [w for _, w in axes])

        # 硬件/云：prob 是 dict[str_bitstring] -> float
        # 组合成 list 形式以复用 _expval_zz_string_from_raw 的 list 分支逻辑
        raw_list = [{"probability": prob}]
        return self._expval_zz_string_from_raw(raw_list, [w for _, w in axes])




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
