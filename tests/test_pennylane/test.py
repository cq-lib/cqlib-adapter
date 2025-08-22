import pennylane as qml
from pennylane import numpy as np


TOKEN = "your_token"
# 如果只使用本地模拟器，可以将login_key设置为None
dev = qml.device('cqlib.device', wires=2, shots=500, cqlib_backend_name="default",login_key = TOKEN)


@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliY(0))
params = np.array([0.5, 0.8], requires_grad=True)
# 初始化设备
def test(cqlib_backend_name):
    dev = qml.device('cqlib.device', wires=2, shots=500, cqlib_backend_name=cqlib_backend_name)
    # 定义一个简单的量子电路
    @qml.qnode(dev)
    def circuit_probs(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])  # 返回4个概率值

    @qml.qnode(dev)
    def circuit_expval(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))  # 返回期望值

    params = np.array([0.5, 0.8])
    probs = circuit_probs(params)
    expval = circuit_expval(params)
    print("概率分布:", probs)
    print("期望值:", expval)

# test("tianyan504")
# test("tianyan_sw")
# test("default")
# 进行简单的优化（模拟量子机器学习）
opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 10
for i in range(steps):
    params = opt.step(circuit, params)

    print(f"步骤 {i + 1}: 参数 = {params}, 期望值 = {circuit(params)}")

# print("最终参数:", params)
# print("最终期望值:", circuit(params))