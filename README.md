# cqlib-adapter 2

`cqlib-adapter` 将 Qiskit、Cirq、PennyLane 和 CUDA-Q 电路接入新版 `cqlib`、`cqlib-tianyan` 与天衍量子计算云平台。

当前版本已提供完整的公共核心以及 Qiskit、PennyLane、Cirq 和 CUDA-Q 适配器。四套适配器共享 cqlib 编译、QCIS、天衍设备/任务和 canonical result 基础设施，并分别提供框架转换、本地模拟器、mock 云闭环及显式启用的真机测试。逐模块验证方法见 `docs/`，总测试入口见 `docs/testing.md`。

## Python 与底座版本

- Python 3.11 及以上；CI 覆盖 3.11–3.13。
- 新版 `cqlib==0.1.0`。
- 新版 `cqlib-tianyan==0.1.0`。

这里必须使用工作区中的新版底座。PyPI 上版本号更高的 `cqlib 1.x` 属于旧产品线，不能替代本项目所需的新版 `0.1.0` API。

## 按框架安装

```bash
pip install "cqlib-adapter[qiskit]"
pip install "cqlib-adapter[cirq]"
pip install "cqlib-adapter[pennylane]"
pip install "cqlib-adapter[cudaq]"
pip install "cqlib-adapter[qiskit,cirq]"
pip install "cqlib-adapter[all]"
```

四个 extra 相互独立。基础包不会自动导入任何量子框架，未安装的框架不会妨碍其他适配器导入。

CUDA-Q 官方当前支持 Linux 和 Apple Silicon macOS；Windows 用户应在 WSL2 中使用。因而 Windows 上安装 `[cudaq]` 或 `[all]` 时会跳过 `cudaq` 依赖，调用 CUDA-Q 适配器时会给出平台和安装提示。

## 最小离线示例

先按照下文安装本地 `cqlib==0.1.0`、`cqlib-tianyan==0.1.0` 和所需框架 extra，然后从仓库根目录运行。以下命令都不读取 API key，也不会创建云任务：

| 适配器 | 最小转换示例 | 最小语义执行示例 |
|---|---|---|
| Qiskit | `python examples/qiskit/01_conversion.py` | `python examples/qiskit/04_grover_simulator.py` |
| PennyLane | `python examples/pennylane/01_conversion.py` | `python examples/pennylane/04_grover_simulator.py` |
| Cirq | `python examples/cirq/01_conversion.py` | `python examples/cirq/04_grover_simulator.py` |
| CUDA-Q（Linux/WSL2） | `python examples/cudaq/01_conversion.py` | `python examples/cudaq/04_grover_simulator.py` |

`01_conversion.py` 展示框架线路到 cqlib/QCIS 的最小路径；`04_grover_simulator.py` 会把框架参考结果与真实 cqlib 本地模拟结果比较，只有语义一致才输出 `PASS`。位序、statevector scaling、换基测量、mock 和显式真机示例见 `examples/README.md` 及各框架子目录的 README。`03_tianyan_cloud.py` 和 `031_tianyan_topology.py` 会在获得明确授权及凭证后创建外部任务，不属于默认示例。

### PennyLane 换基测量

PennyLane 适配器支持有限 shots 下单 wire Pauli X/Y/Z observable 的 `qml.counts`、`qml.sample`、`qml.expval` 和 `qml.var`。转换器会在测量前插入对应的 basis rotation（X 基使用 H，Y 基使用 S†+H，Z 基无需旋转），再把 canonical bit 结果转换成 PennyLane 的 ±1 本征值。同一 wire 同时请求不兼容测量基时会明确报错。最小离线验证：

```bash
python examples/pennylane/08_basis_measurement.py
```

该 example 使用 Pauli X/Y 的已知 +1 本征态验证换基门、cqlib/QCIS 执行和本征值结果；Pauli Z 及 `sample`/`expval`/`var` 的边界由 PennyLane 单元测试覆盖。

## 本地开发环境

```bash
conda env create -f environment-dev.yml
conda activate cqlib-adapter2-dev
```

先分别构建并安装工作区中的新版 Python 绑定（只编译底座，不需要修改 Rust）：

```bash
cd ../cqlib/crates/binding-python
maturin develop --release
cd ../../../cqlib-tianyan/crates/binding-python
maturin develop --release
```

用以下命令确认加载的是本地 `0.1.0` 原生扩展，而不是公开 PyPI 上的旧产品线：

```bash
python -c "from importlib.metadata import version; import cqlib._native, cqlib_tianyan._cqlib_tianyan; print(version('cqlib'), cqlib._native.__file__); print(version('cqlib-tianyan'), cqlib_tianyan._cqlib_tianyan.__file__)"
```

底座安装完成后安装当前项目：

```bash
python -m pip install -e ".[dev]"
```

仅运行不导入真实底座的离线测试时，可使用：

```bash
python -m pip install --no-deps -e .
```

## 常用质量命令

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy cqlib_adapter
python -m pytest -m "not cloud"
python -m pytest --cov=cqlib_adapter -m "not cloud"
python -m build
python -m twine check dist/*
```

真实云测试必须同时提供环境变量并显式选择 `cloud` marker；普通测试和 CI 不读取凭证，也不会创建云任务：

```bash
python -m pytest -m cloud
```

凭证只应通过当前终端的隐藏输入注入环境变量，不能写入源码或命令历史。详见 `SECURITY.md` 和各框架的 `03_tianyan_cloud.py`。

## 公共核心

`cqlib_adapter.common` 现在提供：

- `TranslationBundle`、`TranslationMetadata` 和测量/比特映射契约；
- `CircuitCompiler`：调用新版 cqlib 完成分解、原生门转换、布局/路由，并做 QCIS 往返、门集、拓扑和测量映射校验；
- `NormalizedDevice`：统一设备名称、物理比特 ID、原生门、定向拓扑、运行状态、收费类型和可用性；
- `TianyanConnector`：登录/凭证恢复、设备发现、编译、校准模式选择和任务提交；
- `AdapterJob`：任务 ID、非阻塞状态、等待、超时、批结果重排和结果缓存；
- `ResultConverter`：按 cqlib 小端约定读取天衍结果，恢复框架经典位顺序并生成 counts、probabilities 和 samples。

公共提交层会逐条提交线路，以保证返回 task ID 与编译元数据一一对应。若后续线路提交失败，异常会包含已经创建的 task ID，便于调用者继续查询。

## 项目边界

- `cqlib_adapter.common`：四框架共享的转换、编译、设备、任务和结果基础设施。
- `cqlib_adapter.qiskit`：Qiskit 表面 API。
- `cqlib_adapter.cirq`：Cirq 表面 API。
- `cqlib_adapter.pennylane`：PennyLane Device 表面 API。
- `cqlib_adapter.cudaq`：CUDA-Q target/execution 表面 API。

任何框架模块都不得复制认证、HTTP 或天衍结果解析逻辑；这些能力应集中在公共层并复用 `cqlib-tianyan`。

## 测试原则

- 测试源代码、fixture 和 CI 配置必须提交到 Git。
- `.gitignore` 只忽略缓存、覆盖率、构建结果、真实凭证和云任务输出。
- 每个功能先写成功测试和边界测试，再实现。
- 云平台行为通过 fake/mock 在无密钥 CI 中覆盖，真实云测试单独 opt-in。

详细兼容性和开发边界见 `docs/compatibility.md` 与 `docs/architecture.md`；提交前检查见 `docs/release-checklist.md`。
