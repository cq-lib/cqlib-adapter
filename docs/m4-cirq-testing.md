# Cirq closed-loop validation

## Supported version

The Cirq adapter is implemented and tested against `cirq-core==1.7.0`. The independent package extra remains `cirq-core>=1.7,<2`. Only `cirq-core` is required; vendor packages such as `cirq-google` are not installed by this adapter.

The implementation follows Cirq 1.7's current contracts:

- `cirq.AbstractCircuit`/`cirq.Circuit` for circuit input;
- `cirq.Device` and `cirq.DeviceMetadata` for qubits and NetworkX topology;
- `cirq.Sampler.run` and `run_sweep` for execution;
- `cirq.ResultDict` for standard measurement output;
- `cirq.ParamResolver` and `cirq.to_resolvers` for parameter sweeps.

## Closed-loop architecture

```text
cirq.Circuit
  -> resolve sweep parameters
  -> decompose composite operations
  -> cqlib.Circuit + measurement-key metadata
  -> cqlib compile/layout/routing/native basis
  -> QCIS
  -> cqlib-tianyan submit/wait
  -> CanonicalResult
  -> cirq.ResultDict
```

`TianyanSampler.run()` is inherited from Cirq and delegates to the adapter's `run_sweep()`. Each parameter resolution creates one cqlib compilation artifact and one Tianyan task. The corresponding `CirqExecution` remains available through `sampler.last_executions` with task ID, status, QCIS and cached result.

## Measurement keys and bit order

Each final `cirq.MeasurementGate` must have a unique key. Its qubit order is stored as classical-bit indices in translation metadata. The result converter creates a two-dimensional boolean array for every key, with rows as repetitions and columns exactly matching the measurement gate's qubit arguments.

Cirq converts one measurement row to histogram integers as big-endian bits. For `cirq.measure(q0, q1, q2, key="state")`, state `q0=0, q1=1, q2=1` must be:

```text
cqlib canonical string: 110
Cirq measurement row:   [False, True, True]
Cirq histogram integer: 3
```

The adapter performs this projection globally for mock, local simulator and real Tianyan results. Repeated keys, repeated qubit measurements, confusion maps, mid-circuit operations after measurement and missing measurements are explicitly rejected. Measurement `invert_mask` is supported by inserting the required X before measurement.

## Manual validation order

```powershell
conda activate cqlib-adapter-dev
# Run the remaining commands from the cqlib-adapter repository root.
python -m pip install -e ".[cirq]"
python -c "import cirq; print(cirq.__version__)"
python -m pytest tests\cirq\test_gates.py -q
python -m pytest tests\cirq\test_converter.py -q
python examples\cirq\01_conversion.py
python -m pytest tests\cirq\test_result.py -q
python examples\cirq\05_bit_order_011.py
python -m pytest tests\cirq\test_sampler_device.py -q
python examples\cirq\02_mock_closed_loop.py
python examples\cirq\04_grover_simulator.py
python examples\cirq\06_deutsch_jozsa_simulator.py
python -m pytest tests\cirq\test_topology_example.py -q
python -m pytest tests\integration\test_cirq_cqlib_runtime.py -q
python -m pytest tests\cirq -q
```

Only after all offline checks pass, load the key into the current process environment using the hidden-input PowerShell procedure in `docs/m2-qiskit-testing.md` and verify `DEFAULT_DEVICE` in `examples/cirq/03_tianyan_cloud.py`. Never fill the committed blank key field:

```powershell
python examples\cirq\03_tianyan_cloud.py
python examples\cirq\031_tianyan_topology.py
```

Both scripts check device availability before submission. The topology script additionally compiles and validates every selected physical edge and measurement mapping before creating a task.

The CI-safe live test is skipped by default. Explicit authorization requires:

```powershell
$env:CQLIB_RUN_CLOUD="1"
$env:TIANYAN_DEVICE="tianyan176"
python -m pytest tests\cloud\test_cirq_tianyan_live.py -q -s
```

Clear `TIANYAN_API_KEY` immediately after the test. Do not commit keys or cloud task output.

## Current limitations

The Tianyan surface focuses on finite-repetition sampling and does not expose expectation values, repeated measurement-key records, mid-circuit feedback, noisy channels, non-binary qids or arbitrary matrix gates. The local cqlib sampler additionally exposes `run_statevector()` and restores logical qubit order after non-identity compilation layouts. Measurement-only zero-state circuits are supported. Supported composite operations are decomposed when Cirq provides a valid decomposition; unsupported residual operations raise an actionable conversion error.
