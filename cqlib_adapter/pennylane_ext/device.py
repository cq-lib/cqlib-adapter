# 导入必要的库和模块
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
    """自定义量子设备类，基于CQLib后端实现PennyLane设备接口"""
    
    # 设备元数据
    short_name = 'cqlib.device'  
    pennylane_requires = '>=0.42.1'  # 依赖的PennyLane版本
    version = '0.1.0'  # 设备版本号
    author = ''
    config_filepath = path.join(path.dirname(__file__), "cqlib_config.toml")  # 配置文件路径

    def __init__(self, wires, shots=None, cqlib_backend_name="default", login_key=None):
        """
        初始化CQLib设备
        
        Args:
            wires: 量子比特数量
            shots: 测量次数，None表示使用解析计算
            cqlib_backend_name: CQLib后端名称，默认为"default"
            login_key: 天衍平台登录密钥（如果需要）
        """
        super().__init__(wires=wires, shots=shots)
        self.num_wires = wires  # 量子比特数量
        self.num_shots = shots  # 测量次数
        self.machine_name = cqlib_backend_name  # 后端名称
        
        # 如果不是默认后端，初始化天衍云平台连接
        if cqlib_backend_name != "default":
            self.cqlib_backend = TianYanPlatform(login_key=login_key, machine_name=cqlib_backend_name)

    def name(self):
        """返回设备名称"""
        return 'CQLib Quantum Device'
    
    @classmethod
    def capabilities(cls):
        """
        返回设备能力配置
        
        Returns:
            dict: 包含设备支持的功能和特性的字典
        """
        # 获取父类能力字典
        capabilities = super().capabilities().copy()
        
        # 更新设备特定能力
        capabilities.update(
            model="qubit",  # 量子比特模型（非连续变量）
            supports_inverse_operations=False,  # 是否支持逆操作
            supports_analytic_computation=False,  # 是否支持解析计算
            supports_finite_shots=True,  # 支持有限次测量

            returns_state=False,  # 是否返回量子态
            # 支持的自动微分框架对应的设备
            passthru_devices={
                "autograd": "default.qubit.autograd",
                "tf": "default.qubit.tf",
                "torch": "default.qubit.torch",
                "jax": "default.qubit.jax",
            },
        )
        
        return capabilities
        
    def execute(self, circuits: QuantumScriptOrBatch, execution_config=None):
        """
        在目标后端执行量子线路
        
        Args:
            circuits: 要执行的量子线路（单个或多个）
            execution_config: 执行配置
            
        Returns:
            list: 每个线路的测量结果列表
        """
        # 确保circuits是列表（即使只提供单个线路）
        circuits = [circuits] if isinstance(circuits, qml.tape.QuantumScript) else circuits
        results = []  # 存储所有线路的结果
        
        for circuit in circuits:
            # 在生成 QASM 前，对 PauliX/PauliY 的观测插入换基
            new_ops = list(circuit.operations)
            for meas in circuit.measurements:
                if isinstance(meas, qml.measurements.ExpectationMP):
                    if meas.obs.name == "PauliX":
                        # 在目标比特插入 H 门
                        new_ops.append(qml.Hadamard(wires=meas.obs.wires))
                    elif meas.obs.name == "PauliY":
                        # 插入 S† 和 H
                        new_ops.append(qml.adjoint(qml.S)(wires=meas.obs.wires))
                        new_ops.append(qml.Hadamard(wires=meas.obs.wires))
            # 重新构建量子线路（保持测量不变）
            circuit = qml.tape.QuantumScript(new_ops, circuit.measurements, shots=circuit.shots)
            # 将线路转换为QCIS格式
            qasm_str = circuit.to_openqasm()  # 转换为OpenQASM字符串
            cqlib_cir = qasm2.loads(qasm_str)  # 加载为CQLib线路对象
            cqlib_qcis = cqlib_cir.qcis  # 获取QCIS格式的线路
            
            # 根据后端类型选择执行方式
            # 约定将每种后端的计算结果都调整为dict或dict的list，方便统一处理
            if self._is_tianyan_hardware():
                # 天衍硬件后端：需要编译和提交实验
                compiled_circuit = transpile_qcis(cqlib_qcis, self.cqlib_backend)
                query_id = self.cqlib_backend.submit_experiment(compiled_circuit[0].qcis, num_shots=self.num_shots)
                raw_result = self.cqlib_backend.query_experiment(query_id, readout_calibration=True)
                res = extract_probability(raw_result, num_wires=self.num_wires)
                
            elif self._is_tianyan_simulator():
                # 天衍仿真机后端：直接提交实验
                query_id = self.cqlib_backend.submit_experiment(cqlib_qcis, num_shots=self.num_shots)
                raw_result = self.cqlib_backend.query_experiment(query_id)
                res = extract_probability(raw_result, num_wires=self.num_wires)
                
            else:
                # 本地仿真机：直接执行线路
                # 注意：本地仿真机只接受传入线路对象
                res = self._execute_on_simulator(cqlib_cir)
            
            # 处理测量结果
            circuit_results = self._process_measurements(circuit, res)
            results.append(circuit_results)

        return results

    def _is_tianyan_hardware(self):
        """检查当前后端是否为天衍硬件"""
        return self.machine_name in {"tianyan24", "tianyan504", "tianyan176-2", "tianyan176"}
    
    def _is_tianyan_simulator(self):
        """检查当前后端是否为天衍仿真机"""
        return self.machine_name in {"tianyan_sw", "tianyan_s", "tianyan_tn", "tianyan_tnn", "tianyan_sa", "tianyan_swn"}

    def _execute_on_simulator(self, circuit):
        """在本地仿真机上执行线路"""
        simulator = StatevectorSimulator(circuit)  # 创建态矢量仿真机
        return simulator.sample()  # 返回采样结果

    def _process_measurements(self, circuit: QuantumScript, raw_result: Union[dict, List[dict]]) -> Union[float, np.ndarray, List[Union[float, np.ndarray]]]:
        """
        根据线路的测量操作处理测量结果
        
        Args:
            circuit: PennyLane量子线路
            raw_result: 从后端获取的原始结果（仿真机或硬件）
            
        Returns:
            Union[float, np.ndarray, List]: 测量结果（概率或期望值）
        """
        results = []
        for meas in circuit.measurements:
            if isinstance(meas, qml.measurements.ExpectationMP):
                # 处理期望值测量
                results.append(self._process_expectation(meas, raw_result))
            elif isinstance(meas, qml.measurements.ProbabilityMP):
                # 处理概率测量
                results.append(self._process_probability(meas, raw_result))
            else:
                raise NotImplementedError(f"Measurement type {type(meas).__name__} is not supported")
        
        # 如果只有一个测量结果，直接返回它而不是列表
        return results[0] if len(results) == 1 else results

    def _process_expectation(self, meas, raw_result) -> float:
        """处理期望值测量"""
        # 目前只支持Pauli X Y Z的期望
        if meas.obs.name not in ["PauliZ", "PauliX", "PauliY"]:
            raise NotImplementedError(f"Expectation for {meas.obs.name} is not supported")
        
        # 根据结果类型选择处理方法
        if isinstance(raw_result, list):
            return self.process_results(raw_result)
        elif isinstance(raw_result, dict):
            local_exp = self.process_results_local(raw_result) 
            return local_exp[meas.wires[0]] 
        else:
            raise ValueError(f"Unsupported raw_result type: {type(raw_result)}")

    def _process_probability(self, meas, raw_result) -> np.ndarray:
        """处理概率测量"""
        num_wires = len(meas.wires)  
        probs = np.zeros(2**num_wires) 
        
        if isinstance(raw_result, dict):
            total_shots = sum(raw_result.values())
            for bitstring, count in raw_result.items():
                # 将比特字符串转换为概率数组索引
                index = int(bitstring[::-1], 2)  
                probs[index] = count / total_shots  
                
        elif isinstance(raw_result, list): 
            try:
                # 解析JSON格式的概率分布
                prob_dict = json.loads(raw_result[0]["probability"])
                for bitstring, prob in prob_dict.items():
                    index = int(bitstring[::-1], 2)  
                    probs[index] = prob  
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ValueError(f"Invalid raw_result format: {e}")
        else:
            raise ValueError(f"Unsupported raw_result type: {type(raw_result)}")

        # 检查概率和是否为1（允许小的数值误差）
        if not np.isclose(np.sum(probs), 1.0, rtol=1e-5):
            raise ValueError(f"Probabilities do not sum to 1: {np.sum(probs)}")

        return probs
        
    def process_results(self, raw_result):
        """
        处理硬件或云仿真机的期望值结果
        
        Args:
            raw_result: 原始结果数据
            
        Returns:
            float: PauliZ期望值
        """
        prob_dict = json.loads(raw_result[0]["probability"])
        
        expectation = 0.0
        for state, prob in prob_dict.items():
            if state[0] == "0":  # |0⟩态对应Z的本征值+1
                expectation += prob
            else:                # |1⟩态对应Z的本征值-1
                expectation -= prob
        
        return expectation
        
    def process_results_local(self, raw_result):
        """
        处理本地仿真机的期望值结果
        
        Args:
            raw_result: 原始结果数据
            
        Returns:
            dict: 每个量子比特的PauliZ期望值
        """
        total_shots = sum(raw_result.values())  # 总测量次数
        num_qubits = len(next(iter(raw_result.keys()))) 
        z_expectations = {} 

        for qubit in range(num_qubits):
            n0, n1 = 0, 0 
            for bitstring, count in raw_result.items():
   
                bit = int(bitstring[-qubit - 1])
                if bit == 0:
                    n0 += count
                else:
                    n1 += count
                    
            # 计算Z期望值：(n0 - n1) / total_shots
            z_expectations[qubit] = (n0 - n1) / total_shots

        return z_expectations
        
    def preprocess_transforms(self, execution_config=None):
        """
        定义预处理转换流程
        
        Returns:
            TransformProgram: 预处理转换程序
        """
        program = qml.transforms.core.TransformProgram()
        # 添加设备线数验证
        program.add_transform(qml.devices.preprocess.validate_device_wires, wires=self.wires, name=self.short_name)
        # 添加测量验证
        program.add_transform(qml.devices.preprocess.validate_measurements, name=self.short_name)
        # 添加操作分解（使用supports_operation作为停止条件）
        program.add_transform(qml.devices.preprocess.decompose, stopping_condition=self.supports_operation, name=self.short_name)
        return program

    def supports_operation(self, op):
        """
        检查是否支持特定量子操作
        
        Args:
            op: 量子操作
            
        Returns:
            bool: 如果支持该操作返回True，否则返回False
        """
        # 支持的量子操作集合
        supported_ops = {
            "PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T",
            "RX", "RY", "RZ", "CNOT", "CZ"
        }
        return getattr(op, "name", None) in supported_ops
        
    def __repr__(self):
        """返回设备的字符串表示"""
        return f"<{self.name()} device (wires={self.num_wires}, shots={self.shots})>"


def extract_probability(json_data: List[Dict[str, Union[Dict[str, float], list]]], num_wires: int) -> Dict:
    """
    从JSON数据中提取概率分布
    
    Args:
        json_data: 包含测量结果的JSON数据，期望为包含'dictionary'的列表，
                   其中包含'probability'字段
        num_wires: 线路中的量子比特数（线数）
        
    Returns:
        Dict: 每个量子比特的概率分布
        
    Raises:
        ValueError: 如果JSON数据无效或缺少概率字段
    """
    # 验证输入数据格式
    if not isinstance(json_data, list) or not json_data:
        raise ValueError("json_data must be a non-empty list")
    if not isinstance(json_data[0], dict):
        raise ValueError("json_data[0] must be a dictionary")
    if "probability" not in json_data[0]:
        raise ValueError("probability field missing in json_data[0]")

    try:
        # 提取并解析概率字段
        prob_dict = json_data[0]["probability"]
        if isinstance(prob_dict, str):
            prob_dict = json.loads(prob_dict)  # 如果概率字段是字符串，解析为字典
        if not isinstance(prob_dict, dict):
            raise ValueError("probability field must be a dictionary")
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid probability field format: {e}")

    return prob_dict