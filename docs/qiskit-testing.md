# Qiskit 闭环与逐模块测试指南

## 1. 闭环和代码位置

Qiskit 数据流：Qiskit `QuantumCircuit` → `qiskit_to_cqlib` → `cqlib.Circuit`
→ `cqlib.compile` → QCIS → `cqlib-tianyan` → `ExecutionResult`
→ `CanonicalResult` → Qiskit `Result`。

- 原生门：`cqlib_adapter/qiskit/gates.py`
- 线路转换及 QCIS：`cqlib_adapter/qiskit/converter.py`
- Target/CouplingMap：`cqlib_adapter/qiskit/target.py`
- BackendV2：`cqlib_adapter/qiskit/backend.py`
- JobV1：`cqlib_adapter/qiskit/job.py`
- Result：`cqlib_adapter/qiskit/result.py`
- SamplerV2：`cqlib_adapter/qiskit/sampler.py`

## 2. 适配环境

声明范围：Python `>=3.11`、Qiskit `>=2.5,<3`、cqlib `==0.1.0`、
cqlib-tianyan `==0.1.0`。

本次实际验证：Python 3.11.15 64 位、Qiskit 2.5.0，以及本地 Rust 编译生成的
`cqlib._native.pyd` 和 `cqlib_tianyan._cqlib_tianyan.pyd`。

```powershell
conda activate cqlib-adapter-dev
# Run the remaining commands from the cqlib-adapter repository root.
python -c "import qiskit; print(qiskit.__version__)"
python -c "import cqlib._native as n; print(n.__file__)"
python -c "import cqlib_tianyan._cqlib_tianyan as n; print(n.__file__)"
python -m pip check
```

重新安装适配器和 Qiskit 可选依赖：

```powershell
python -m pip install -e ".[qiskit,dev]"
```

## 3. 原生门

测试代码：`tests/qiskit/test_gates.py`。

覆盖 X2P/X2M、Y2P/Y2M、XY/XY2P/XY2M、RXY、FSIM、参数绑定、逆门和
QCIS 名称映射。

```powershell
python -m pytest tests/qiskit/test_gates.py -vv
```

## 4. 线路转换和 cqlib 编译

测试代码：`tests/qiskit/test_converter.py`；手工脚本：
`examples/qiskit/01_conversion.py`。

覆盖标准门及原生门到真实 `cqlib.Circuit`、参数和经典寄存器、Bell 线路到原生
QCIS、受支持的反向转换，以及未绑定参数、未知门、测量后门和重复经典位。

```powershell
python -m pytest tests/qiskit/test_converter.py -vv
python examples/qiskit/01_conversion.py
```

输出 QCIS 应包含 `CZ`、`RZ`、半旋转门和两个 `M`。

## 5. Target、门分解和拓扑

测试代码：`tests/qiskit/test_target.py`。

测试使用真实 `cqlib.device.Device` 创建 Qiskit `Target`，并调用 Qiskit 自己的
`transpile`，检查原生门、可用比特、有向拓扑、H/CX 分解、非相邻比特路由、
稀疏物理 ID 和未知门异常。

```powershell
python -m pytest tests/qiskit/test_target.py -vv
```

## 6. Backend、Job、Result 和 mock 云闭环

测试代码：`tests/qiskit/test_backend_job_result.py`；mock：
`cqlib_adapter/qiskit/testing.py`；手工脚本：`examples/qiskit/02_mock_closed_loop.py`。

mock 仅替换网络平台和等待过程。设备和返回值仍分别使用真实 `cqlib.Device` 和
`cqlib.ExecutionResult`，编译和 QCIS 生成使用真实 `_native.pyd`。

```powershell
python -m pytest tests/qiskit/test_backend_job_result.py -vv
python examples/qiskit/02_mock_closed_loop.py
```

关键预期：任务 ID 为 `mock-task-1`；`job.qcis[0]` 是实际提交内容；
`result` 是标准 Qiskit `Result`；counts 为 `{'00': 70, '11': 30}`；
memory 长度为 100；概率可从 `result.data()['probabilities']` 读取。

## 7. 一次运行 Qiskit 和完整回归

```powershell
python -m pytest tests/qiskit -vv
python -m pytest tests/integration/test_cqlib_runtime.py tests/integration/test_cqlib_tianyan_runtime.py tests/qiskit -vv
python -m pytest -q
python -m pytest --cov=cqlib_adapter --cov-report=term-missing -q
python -m ruff check .
python -m ruff format --check .
python -m mypy cqlib_adapter
```

## 8. 真实天衍云测试

真实测试默认跳过，避免误提交或产生费用。pytest 代码在
`tests/cloud/test_qiskit_tianyan_live.py`，独立示例在
`examples/qiskit/03_tianyan_cloud.py`。

```powershell
$secureKey = Read-Host "Tianyan API key（粘贴后按 Enter）" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:TIANYAN_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    $secureKey.Dispose()
}
$env:TIANYAN_DEVICE = "负责人提供的设备名称"
$env:TIANYAN_TEST_SHOTS = "100"
$env:TIANYAN_TEST_TIMEOUT = "300"
```

先查询设备，不提交：

```powershell
python -c "from cqlib_adapter.qiskit import TianyanBackend; import os; b=TianyanBackend.login(os.environ['TIANYAN_API_KEY'], os.environ['TIANYAN_DEVICE'], save_credentials=False); print(b.status()); print(sorted(b.target.operation_names)); print(b.target.build_coupling_map())"
```

确认设备和费用后，显式授权提交：

```powershell
$env:CQLIB_RUN_CLOUD = "1"
python -m pytest tests/cloud/test_qiskit_tianyan_live.py -vv -s
```

结束后执行 `Remove-Item Env:TIANYAN_API_KEY`。不要把真实密钥写入源码、配置文件、命令文本或日志，也不要提交任何云任务结果或凭证文件。

或运行：

```powershell
python examples/qiskit/03_tianyan_cloud.py
```

提交后检查 `job.task_ids`、`job.status()`、`job.qcis[0]`、
`result.get_counts()`、`result.get_memory()` 和
`result.data()['probabilities']`。

## 9. 当前限制

线路参数必须先绑定，测量必须是最终操作。适配器明确拒绝动态控制流、条件门、
`initialize`、未知自定义门、测量后的量子门和未绑定参数，避免静默误编译。
