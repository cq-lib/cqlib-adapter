# Cqlib Adapter

Cqlib Adapter 是一个用于将 Qiskit 量子电路转换为 Cqlib 兼容格式的工具，它允许用户通过 TianYan 平台访问实际的量子计算机和模拟器。

## 安装

要安装 Cqlib Adapter，请使用 pip:

```bash
pip install cqlib-adapter
```

## 特性

- **QCIS 门支持**：支持多种量子门操作，包括 X2P, X2M, Y2P, Y2M, XY2P, XY2M 和 Rxy 门。
- **TianYan 平台集成**：可以通过 TianYan 提供商轻松获取并使用特定的后端设备。
- **采样器模式**：支持在指定的后端上运行量子电路，并获取结果。

## 使用示例

### 创建量子电路

您可以使用 Qiskit 创建一个量子电路，并使用 Cqlib Adapter 将其转换为 Cqlib 格式。

```python
from qiskit import QuantumCircuit
from cqlib_adapter.qiskit_ext.adapter import to_cqlib

# 创建一个简单的量子电路
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# 转换为 Cqlib 格式
cqlib_circuit = to_cqlib(qc)
```

#### 后端模式

1. **初始化 TianYan 提供商**

   ```python
   from cqlib_adapter.qiskit_ext.tianyan_provider import TianYanProvider

   provider = TianYanProvider(token='your_token')
   ```

2. **获取特定后端**

   ```python
   backend = provider.backend(name='tianyan176-2')
   ```

3. **在后端上运行电路**

   ```python
   job = backend.run(qc)
   ```

4. **获取并打印结果**

   ```python
   result = job.result()
   print(result.get_counts())
   ```

#### 采样器模式

1. **初始化 TianYan 提供商**

   ```python
   provider = TianYanProvider(token='your_token')
   ```

2. **获取特定后端**

   ```python
   backend = provider.backend(name='tianyan24')
   ```

3. **在后端上运行电路**

   ```python
   from cqlib_adapter.qiskit_ext.sampler import TianYanSampler

   sampler = TianYanSampler(backend)
   job = sampler.run([qc])
   ```

4. **获取并打印结果**

   ```python
   result = job.result()
   print(result)
   ```

注意：`c0` 是默认的寄存器名称。

## 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 贡献

欢迎贡献代码和反馈。请提交 PR 或 issue 到 [Gitee 仓库](https://gitee.com/cq-lib/cqlib-adapter)。