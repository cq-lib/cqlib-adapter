# Architecture

The adapter is split into a framework-independent core and four thin framework surfaces.

```text
Qiskit / Cirq / PennyLane / CUDA-Q
                 |
       framework translation bundle
                 |
          cqlib.Circuit
                 |
       cqlib compiler + device
                 |
              QCIS
                 |
       cqlib-tianyan backend
                 |
       cqlib ExecutionResult
                 |
       framework-native result
```

The common layer implements the complete framework-independent path. Framework adapters only translate circuits into a `TranslationBundle` and project `CanonicalResult` into framework-native objects. They must not duplicate authentication, submission, polling, bit ordering, or error normalization.

## Common module responsibilities

| Module | Responsibility |
|---|---|
| `common.circuit` | Logical/physical qubit mapping, classical destinations, translation metadata, compiled measurement binding. |
| `common.compiler` | Lazy cqlib boundary, compilation options, target basis and topology checks, QCIS serialize/parse round trip. |
| `common.device` | Tianyan status/toll normalization, physical qubit IDs, native gates and directed couplings. |
| `common.platform` | Tianyan authentication, discovery, availability guard, compilation and per-circuit submission. |
| `common.job` | Task IDs, ready/partial/done/error snapshots, global timeout budget, response reordering and caching. |
| `common.result` | Cloud status validation, counts/probability validation, endian conversion and deterministic samples. |

## Measurement and bit-order contract

Framework translators assign stable logical qubit IDs and final classical-bit destinations. cqlib compilation may change physical qubits, so `CircuitCompiler` binds final measurement operations to physical IDs after routing.

cqlib `ExecutionResult` outcomes are little-endian with respect to `result.qubits`: the rightmost bit belongs to `result.qubits[0]`. The common converter first builds a physical-qubit bit map, applies the compiled measurement bindings, then emits canonical MSB-left classical strings. A canonical sample row is ordered `c0, c1, ...`; framework layers may regroup or reformat those bits without reinterpreting the cloud payload.

## Submission invariant

The common layer submits each compiled circuit independently. This avoids relying on an undocumented batch response order and makes each returned task ID map to exactly one `CompilationArtifact`. The resulting `AdapterJob` still presents them as one ordered batch. If a later submission fails, `AdapterSubmissionError` includes the already-created task IDs.

## Known bottom-layer limitation

The current cqlib-tianyan Python `device_config()` conversion reconstructs a native cqlib `Device` but does not currently copy every Rust-side calibration/invalid-qubit property. `NormalizedDevice.properties_complete` is therefore `False`. The adapter never fabricates missing calibration data; framework targets should expose the available topology and gates and treat omitted detailed properties as unknown.
