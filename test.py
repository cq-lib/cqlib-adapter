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



# 创建设备
# dev = qml.device("cqlib.device",wires=[0,1,2])
dev = qml.device("default.qubit",wires=[0,1,2],shots = 1000)
@qml.qnode(dev, interface='auto',autograph = False)
def circuit():
    qml.H(0)
    qml.CNOT(wires=[0, 2])
    # 计算 Z⊗Z⊗Z 的期望值
    # return qml.expval(qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(2))
    # return qml.probs()
    return qml.sample()

# 执行电路
result = circuit()
print(f"Joint expectation value: {result}")