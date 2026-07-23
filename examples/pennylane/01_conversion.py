"""PennyLane QuantumScript -> real cqlib Circuit -> native QCIS."""

from pennylane import CNOT, Hadamard, counts
from pennylane.tape import QuantumScript

from cqlib_adapter.common import CompilationOptions
from cqlib_adapter.pennylane import compile_pennylane_circuit, pennylane_to_cqlib

NATIVE_BASIS = (
    "RZ",
    "X2P",
    "X2M",
    "Y2P",
    "Y2M",
    "XY2P",
    "XY2M",
    "CZ",
    "GPHASE",
)


def main() -> None:
    tape = QuantumScript(
        [Hadamard(0), CNOT((0, 1))],
        [counts(wires=(0, 1))],
        shots=100,
    )
    bundle = pennylane_to_cqlib(tape, wire_order=(0, 1))
    print("cqlib operations:")
    for operation in bundle.circuit.operations:
        print(" ", operation)

    artifact = compile_pennylane_circuit(
        tape,
        wire_order=(0, 1),
        options=CompilationOptions(target_basis=NATIVE_BASIS, seed=19),
    )
    print("\nTianyan-compatible QCIS:")
    print(artifact.qcis)
    print("measurement mapping:", artifact.measurements)

    if "CZ Q0 Q1" not in artifact.qcis or artifact.qcis.count("M Q") != 2:
        raise AssertionError("PennyLane circuit was not lowered to expected native QCIS")
    print("PASS: PennyLane -> cqlib -> QCIS conversion succeeded.")


if __name__ == "__main__":
    main()
