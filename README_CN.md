# cqlib-adapter

[English](README.md)

`cqlib-adapter` 将 Qiskit、Cirq、PennyLane 和 CUDA-Q 电路接入新版 `cqlib`、`cqlib-tianyan` 与天衍量子计算云平台。

项目提供共享的 cqlib 编译、QCIS、天衍设备/任务和 canonical result 基础设施，以及 Qiskit、Cirq、PennyLane、CUDA-Q 适配器。每套适配器包含框架转换、mock 云闭环和显式启用的真机测试。完整测试说明见 [docs/testing.md](docs/testing.md)，CUDA-Q 说明见 [docs/cudaq-testing.md](docs/cudaq-testing.md)。

## 运行要求

- Python 3.11 及以上；CI 在 Linux、Windows、macOS 上测试下限与最新版本（3.11 和 3.14）。
- 本项目需要新版 `cqlib` 与 `cqlib-tianyan` 原生绑定，两者已发布到 PyPI，安装适配器时作为项目依赖自动拉取；兼容区间以 [pyproject.toml](pyproject.toml) 的依赖声明为准。

## 按框架安装

```bash
pip install "cqlib-adapter[qiskit]"
pip install "cqlib-adapter[cirq]"
pip install "cqlib-adapter[pennylane]"
pip install "cqlib-adapter[cudaq]"
pip install "cqlib-adapter[all]"
```

四个 extra 相互独立；基础包不会自动导入量子框架。CUDA-Q 支持 Linux 和 Apple Silicon macOS；Windows 用户应通过 WSL2 使用。Windows 上的 `[cudaq]`/`[all]` 会跳过 CUDA-Q 依赖，并在调用时给出安装提示。

## 最小离线示例

先安装 `cqlib`、`cqlib-tianyan` 和所需框架 extra（安装适配器时 pip 会从 PyPI 自动拉取这两个原生包），再从仓库根目录运行。下面的命令不读取 API key，也不创建云任务：

| 适配器 | 转换 | mock 闭环 |
|---|---|---|
| Qiskit | `python examples/qiskit/01_conversion.py` | `python examples/qiskit/02_mock_closed_loop.py` |
| PennyLane | `python examples/pennylane/01_conversion.py` | `python examples/pennylane/02_mock_closed_loop.py` |
| Cirq | `python examples/cirq/01_conversion.py` | `python examples/cirq/02_mock_closed_loop.py` |
| CUDA-Q（Linux/WSL2） | `python examples/cudaq/01_conversion.py` | `python examples/cudaq/02_mock_closed_loop.py` |

`01_conversion.py` 展示框架线路到 cqlib/QCIS 的最小路径；`02_mock_closed_loop.py` 用预置 counts 的 mock 云传输跑通 Backend/Sampler/Device 的完整提交链路，验证传输、任务与结果结构契约，而不是量子语义。真机示例见 [examples/README.md](examples/README.md) 和各框架目录 README。真机 `03_tianyan_cloud.py` 与 `031_tianyan_topology.py` 需要明确授权和当前终端中的凭证。

CUDA-Q 的正式路径直接读取 kernel/builder 的 Quake MLIR 并构造 `cqlib.Circuit`，不依赖 OpenQASM 2，也不在失败时回退 QASM。`cudaq_to_openqasm()` 只是用户主动调用的诊断导出工具。固定宽度多 `qalloc`、参数化 decorator、简单标量 builder、可静态求值循环、终端 `mx`/`my`/`mz` 均受支持；无显式测量时会补充全量 `mz`，动态线路结构会明确报错。

### PennyLane 换基测量

有限 shots 下，PennyLane 支持单 wire Pauli X/Y/Z observable 的 `qml.counts`、`qml.sample`、`qml.expval` 和 `qml.var`。转换器测量前插入换基门：X 基为 H，Y 基为 S†+H，Z 基不变，并将 canonical bit 结果转换为 PennyLane 的 ±1 本征值。同一 wire 请求不兼容测量基会明确报错。换基门、本征值转换及 `sample`/`expval`/`var` 边界均由单元测试覆盖。

### PennyLane 认证与运行参数

`TianyanDevice.login()` 与 `TianyanDevice.from_credentials()` 将认证参数和 Device 运行参数分开：

```python
device = TianyanDevice.login(
    api_key,
    "tianyan176",
    login_options={"domain": "https://platform.example"},
    timeout=120,
    poll_interval=5,
    calibration="auto",
    compilation_mode="normal",
    seed=7,
)
```

已保存凭证使用 `credential_options={"credentials_path": ...}` 调用 `from_credentials()`。两个 mapping 接受 `domain`、`auto_refresh`、`credentials_path`；`credential_options` 还可接受 `save_credentials`。未知字段在登录或加载前报错。`login()` 的 `save_credentials` 必须作为显式参数传入，不能放在 `login_options`。

## 本地开发环境

```bash
conda env create -f environment-dev.yml
conda activate cqlib-adapter-dev
```

`environment-dev.yml` 创建 Python、可选框架和质量工具环境。原生 `cqlib` 与 `cqlib-tianyan` 绑定已发布到 PyPI，并为受支持平台提供预编译 wheel，无需再从源码编译。

以可编辑模式安装适配器，pip 会把项目依赖 `cqlib` 与 `cqlib-tianyan` 一并从 PyPI 拉取：

```bash
python -m pip install -e ".[dev]"
```

确认加载的是 PyPI 原生扩展，版本应满足 [pyproject.toml](pyproject.toml) 声明的兼容区间：

```bash
python -c "from importlib.metadata import version; import cqlib._native, cqlib_tianyan._cqlib_tianyan; print(version('cqlib'), cqlib._native.__file__); print(version('cqlib-tianyan'), cqlib_tianyan._cqlib_tianyan.__file__)"
```

`dev` extra 是文档约定的 Windows/macOS 开发测试面，包含质量工具与 Qiskit、Cirq、PennyLane；Linux/WSL 在此基础上显式追加 CUDA-Q：

```bash
# Windows/macOS：质量工具、Qiskit、Cirq、PennyLane
python -m pip install -e ".[dev]"
# Linux/WSL：另加 CUDA-Q
python -m pip install -e ".[dev,cudaq]"
```

## 测试与质量检查

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy cqlib_adapter
# Windows：CUDA-Q 由 WSL/Linux 独立验证。
python -m pytest -m "not cloud and not cudaq"
python -m pytest --cov=cqlib_adapter --cov-config=coverage-windows.ini --cov-report=term-missing -m "not cloud and not cudaq"
# Linux/WSL（安装 .[dev,cudaq] 后）：
python -m pytest -m "not cloud"
python -m pytest --cov=cqlib_adapter --cov-report=term-missing -m "not cloud"
python -m build
python -m twine check dist/*
```

普通测试和 CI 不读取凭证，也不会创建云任务。真机测试必须显式选择 `cloud` marker，并通过当前终端的隐藏输入设置环境变量。不要将 API key 写入源码、配置、命令历史或日志；结束后清除环境变量。各真机 example 有相应操作说明。

## 公共核心与边界

`cqlib_adapter.common` 提供 `TranslationBundle`/`TranslationMetadata`、`CircuitCompiler`、`NormalizedDevice`、`TianyanConnector`、`AdapterJob` 和 `ResultConverter`。编译器对测量、屏障等指令不要求耦合边，只有实际双比特量子门才按对称性或控制/目标方向检查拓扑。`timeout` 和 `poll_interval` 必须为有限正数；`NaN`、无穷、布尔和非数值会在提交或等待前报错。

公共提交层逐条提交线路，因此 task ID 与编译元数据一一对应；后续提交失败时异常会保留已创建的 task ID。框架模块不得复制认证、HTTP 或天衍结果解析逻辑，必须复用公共层和 `cqlib-tianyan`。

## 发布包边界

源码发布包有意保持最小：包含包源码、`py.typed`、LICENSE 和基础安装/导入 smoke tests；文档、示例和完整测试套件留在 Git 工作区，由 CI 验证。

```bash
python -m build
python -m twine check dist/*
```

提交或发布前删除 `.coverage`、缓存、日志、`build/`、`dist/` 和 `*.egg-info/`。这些可再生成文件已被 Git 忽略；完整发布检查见 [docs/release-checklist.md](docs/release-checklist.md)。

## 测试原则

- 测试、fixture 和 CI 配置必须提交到 Git。
- `.gitignore` 仅忽略缓存、覆盖率、构建结果、真实凭证和云任务输出。
- 每个功能先写成功与边界测试。
- 云平台行为通过 fake/mock 在无密钥 CI 中覆盖；真机测试独立 opt-in。

详细兼容性与架构见 [docs/compatibility.md](docs/compatibility.md) 和 [docs/architecture.md](docs/architecture.md)。
