import pennylane as qml
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScriptOrBatch
import numpy as np
import cqlib
from cqlib import TianYanPlatform
import json
from cqlib.utils import qasm2
from os import path
from cqlib.mapping import transpile_qcis
from cqlib.simulator import StatevectorSimulator
from typing import Union, List, Dict
from pennylane.tape import QuantumScript



class CQLibDevice(Device):
    short_name = 'cqlib.device'
    pennylane_requires = '0.42.1'
    version = '0.1.0'
    author = 'Ky'
    config_filepath = path.join(path.dirname(__file__), "cqlib_config.toml")

    def __init__(self, wires, shots=None, cqlib_backend_name="default", login_key = None):
        super().__init__(wires=wires, shots=shots)
        self.num_wires = wires
        self.num_shots = shots
        self.machine_name = cqlib_backend_name
        if cqlib_backend_name != "default":
            self.cqlib_backend = TianYanPlatform(login_key=login_key, machine_name=cqlib_backend_name)
    def name(self):
        return 'CQLib Quantum Device'
    @classmethod
    def capabilities(cls):
        # 获取父类能力字典
        capabilities = super().capabilities().copy()  # 必须创建副本以避免修改父类属性
        
        # 更新设备特定能力
        capabilities.update(
            model="qubit",  # 或 "cv" (连续变量)
            supports_inverse_operations=True,
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
        """Execute quantum circuits on the target backend.
        
        Args:
            circuits: Quantum circuit(s) to execute
            execution_config: Execution configuration
            
        Returns:
            List of results corresponding to each circuit's measurements
        """
        # Ensure circuits is a list even if single circuit is provided
        circuits = [circuits] if isinstance(circuits, qml.tape.QuantumScript) else circuits
        results = []
        
        for circuit in circuits:
            # Convert circuit to QCIS format
            qasm_str = circuit.to_openqasm()
            cqlib_cir = qasm2.loads(qasm_str)
            cqlib_qcis = cqlib_cir.qcis
            print(qasm_str)
            
            # 根据后端类型选择执行方式
            # 此处约定将每一种后端的计算结果都调整为一个dict或者dict的list，方便统一处理
            if self._is_tianyan_hardware():
                compiled_circuit = transpile_qcis(cqlib_qcis, self.cqlib_backend)
                query_id = self.cqlib_backend.submit_experiment(compiled_circuit[0].qcis, num_shots=self.num_shots)
                raw_result = self.cqlib_backend.query_experiment(query_id, readout_calibration=True)
                res = extract_probability(raw_result, num_wires=self.num_wires)
                # print(res)

            elif self._is_tianyan_simulator():
                query_id = self.cqlib_backend.submit_experiment(cqlib_qcis, num_shots=self.num_shots)
                raw_result = self.cqlib_backend.query_experiment(query_id)
                res = extract_probability(raw_result, num_wires=self.num_wires)
                # print(res)
            else:
                # caution: 本地模拟器只接受线路对象，不是qcis串
                res = self._execute_on_simulator(cqlib_cir)
                # print(res)
            
            # Process measurements
            circuit_results = self._process_measurements(circuit, res)
            results.append(circuit_results)

        return results

    def _is_tianyan_hardware(self):
        """Check if the current backend is Tianyan hardware."""
        return self.machine_name in {"tianyan24", "tianyan504", "tianyan176-2", "tianyan176"}
    
    def _is_tianyan_simulator(self):
        """Check if the current backend is Tianyan simulator."""
        return self.machine_name in {"tianyan_sw", "tianyan_s", "tianyan_tn", "tianyan_tnn", "tianyan_sa", "tianyan_swn"}

    def _execute_on_simulator(self, circuit):
        """Execute circuit on local simulator."""

        # simulator = SimpleSimulator(circuit, device="cpu")
        simulator = StatevectorSimulator(circuit)
        return simulator.sample()

    def _process_measurements(self, circuit: QuantumScript, raw_result: Union[dict, List[dict]]) -> Union[float, np.ndarray, List[Union[float, np.ndarray]]]:
        """
        Process measurements based on circuit's measurement operations.

        Args:
            circuit: PennyLane quantum circuit.
            raw_result: Raw result from the backend (simulator or hardware).

        Returns:
            Measurement results (probabilities or expectations).
        """
        results = []
        for meas in circuit.measurements:
            if isinstance(meas, qml.measurements.ExpectationMP):
                results.append(self._process_expectation(meas, raw_result))
            elif isinstance(meas, qml.measurements.ProbabilityMP):
                results.append(self._process_probability(meas, raw_result))
            else:
                raise NotImplementedError(f"Measurement type {type(meas).__name__} is not supported")
        return results[0] if len(results) == 1 else results


    def _process_expectation(self, meas, raw_result) -> float:
        """Process expectation value measurement."""
        # 目前只支持 PauliZ 的期望
        if meas.obs.name != "PauliZ":
            raise NotImplementedError(f"Expectation for {meas.obs.name} is not supported")
        
        # 使用 local 或 hardware 解析
        if isinstance(raw_result, list):
            return self.process_results(raw_result)
        elif isinstance(raw_result, dict):
            local_exp = self.process_results_local(raw_result)
            return local_exp[meas.wires[0]]
        else:
            raise ValueError(f"Unsupported raw_result type: {type(raw_result)}")


    def _process_probability(self, meas, raw_result) -> np.ndarray:
        """Process probability measurement."""
        num_wires = len(meas.wires)
        probs = np.zeros(2**num_wires)
        
        if isinstance(raw_result, dict):  # 本地 simulator
            total_shots = sum(raw_result.values())
            for bitstring, count in raw_result.items():
                index = int(bitstring[::-1], 2)
                probs[index] = count / total_shots
        elif isinstance(raw_result, list):  # 硬件或云 simulator
            try:
                prob_dict = json.loads(raw_result[0]["probability"])
                for bitstring, prob in prob_dict.items():
                    index = int(bitstring[::-1], 2)
                    probs[index] = prob
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ValueError(f"Invalid raw_result format: {e}")
        else:
            raise ValueError(f"Unsupported raw_result type: {type(raw_result)}")

        # 检查概率和
        if not np.isclose(np.sum(probs), 1.0, rtol=1e-5):
            raise ValueError(f"Probabilities do not sum to 1: {np.sum(probs)}")

        return probs
    def process_results(self, raw_result):
        # 解析概率分布（例如：{"00":0.788, "11":0.044, ...}）
        prob_dict = json.loads(raw_result[0]["probability"])
        
        expectation = 0.0
        for state, prob in prob_dict.items():
            # 计算第一个量子比特的 Z 期望值
            if state[0] == "0":  # |0⟩ 对应 Z 的本征值 +1
                expectation += prob
            else:                # |1⟩ 对应 Z 的本征值 -1
                expectation -= prob
        
        return expectation
    def process_results_local(self, raw_result):
        total_shots = sum(raw_result.values())

        # 初始化每个量子比特的计数
        z_expectations = {}
        num_qubits = len(next(iter(raw_result.keys())))  # 获取比特串长度（这里是2个量子比特）

        for qubit in range(num_qubits):
            n0, n1 = 0, 0
            for bitstring, count in raw_result.items():
                # 检查当前量子比特的值（注意比特串顺序可能是小端序）
                bit = int(bitstring[-qubit - 1])  # 假设右侧是qubit0，左侧是qubit1
                if bit == 0:
                    n0 += count
                else:
                    n1 += count
            z_expectations[qubit] = (n0 - n1) / total_shots

        return z_expectations
        
    def preprocess_transforms(self, execution_config=None):
        program = qml.transforms.core.TransformProgram()
        program.add_transform(qml.devices.preprocess.validate_device_wires, wires=self.wires, name=self.short_name)
        program.add_transform(qml.devices.preprocess.validate_measurements, name=self.short_name)
        program.add_transform(qml.devices.preprocess.decompose, stopping_condition=self.supports_operation, name=self.short_name)
        return program

    def supports_operation(self, op):
        supported_ops = {
            "PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T",
            "RX", "RY", "RZ", "CNOT", "CZ"
        }
        return getattr(op, "name", None) in supported_ops
        
    def __repr__(self):
        return f"<{self.name()} device (wires={self.num_wires}, shots={self.shots})>"


def extract_probability(json_data: List[Dict[str, Union[Dict[str, float], list]]], num_wires: int) -> Dict:
    """
    Extract probability distribution from JSON data.

    Args:
        json_data: JSON data containing measurement results, expected as a list with a dictionary
                   containing a 'probability' field.
        num_wires: Number of qubits (wires) in the circuit.

    Returns:
        Dict: Probability distribution for each qubit.
    Raises:
        ValueError: If JSON data is invalid or probability field is missing/invalid.
    """
    if not isinstance(json_data, list) or not json_data:
        raise ValueError("json_data must be a non-empty list")
    if not isinstance(json_data[0], dict):
        raise ValueError("json_data[0] must be a dictionary")
    if "probability" not in json_data[0]:
        raise ValueError("probability field missing in json_data[0]")

    try:
        prob_dict = json_data[0]["probability"]
        if isinstance(prob_dict, str):
            prob_dict = json.loads(prob_dict)
        if not isinstance(prob_dict, dict):
            raise ValueError("probability field must be a dictionary")
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid probability field format: {e}")

    return prob_dict