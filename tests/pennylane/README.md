# PennyLane tests

The suite targets PennyLane `>=0.44,<1` and covers:

- QCIS-native Operation matrices, unitarity and adjoints;
- QuantumScript/cqlib conversion, native compilation, QCIS and reverse conversion;
- PennyLane wire labels, measurement metadata and canonical bit-order restoration;
- counts, probabilities, samples, multiple measurements and `all_outcomes`;
- QNode execution through a mock Tianyan transport using real cqlib Device/ExecutionResult objects;
- finite shots, shot-vector rejection, unavailable devices, capacity and malformed requests.

Run one layer at a time:

```powershell
python -m pytest tests\pennylane\test_operations.py -q
python -m pytest tests\pennylane\test_converter.py -q
python -m pytest tests\pennylane\test_result.py -q
python -m pytest tests\pennylane\test_device.py -q
python -m pytest tests\pennylane -q
```

The real test is opt-in and requires `CQLIB_RUN_CLOUD=1`, `TIANYAN_API_KEY`, and `TIANYAN_DEVICE`.
