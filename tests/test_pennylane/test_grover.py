import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


def create_grover_circuit(num_qubits, target_state, shots=1000):
    """Create a generic Grover algorithm circuit for any number of qubits and target state.

    Args:
        num_qubits: Number of qubits in the quantum circuit.
        target_state: The target state to search for (decimal representation).
        shots: Number of measurement shots.

    Returns:
        A Grover circuit function that takes iterations as parameter.
    """
    # dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
    dev = qml.device("cqlib.device", wires=num_qubits, shots=shots, cqlib_backend_name="default")

    def grover_oracle():
        """Oracle that marks the specified target state."""
        target_binary = format(target_state, f"0{num_qubits}b")

        # Apply X gates to qubits that should be 0 in the target state
        for i, bit in enumerate(target_binary):
            if bit == "0":
                qml.PauliX(wires=i)

        # Apply multi-controlled Z gate to flip the phase
        if num_qubits == 1:
            qml.PauliZ(wires=0)
        else:
            qml.Hadamard(wires=num_qubits - 1)
            qml.ctrl(qml.PauliZ, control=list(range(num_qubits - 1)))(wires=num_qubits - 1)
            qml.Hadamard(wires=num_qubits - 1)

        # Undo the X gates
        for i, bit in enumerate(target_binary):
            if bit == "0":
                qml.PauliX(wires=i)

    def diffusion_operator():
        """Diffusion operator for amplitude amplification."""
        # Apply Hadamard gates to all qubits
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)

        # Apply X gates to all qubits
        for wire in range(num_qubits):
            qml.PauliX(wires=wire)

        # Apply multi-controlled Z gate
        if num_qubits == 1:
            qml.PauliZ(wires=0)
        else:
            qml.Hadamard(wires=num_qubits - 1)
            qml.ctrl(qml.PauliZ, control=list(range(num_qubits - 1)))(wires=num_qubits - 1)
            qml.Hadamard(wires=num_qubits - 1)

        # Apply X gates again
        for wire in range(num_qubits):
            qml.PauliX(wires=wire)

        # Apply Hadamard gates again
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)

    @qml.qnode(dev)
    def grover_circuit(iterations=1):
        """Grover circuit with specified number of iterations.

        Args:
            iterations: Number of Grover iterations to perform.

        Returns:
            Probability distribution over all computational basis states.
        """
        # Initialization: Create uniform superposition
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)

        # Grover iterations
        for _ in range(iterations):
            grover_oracle()  # Mark target state
            diffusion_operator()  # Amplify amplitude

        return qml.probs(wires=range(num_qubits))

    return grover_circuit


def calculate_optimal_iterations(num_qubits, num_solutions=1):
    """Calculate optimal number of Grover iterations.

    Args:
        num_qubits: Number of qubits in the quantum circuit.
        num_solutions: Number of target solutions (default: 1).

    Returns:
        Optimal number of Grover iterations.
    """
    search_space_size = 2**num_qubits
    optimal_iterations = int(np.floor((np.pi / 4) * np.sqrt(search_space_size / num_solutions)))
    return optimal_iterations


def run_grover_search(num_qubits, target_state, shots=1000, max_iterations=None):
    """Run complete Grover search with analysis.

    Args:
        num_qubits: Number of qubits in the quantum circuit.
        target_state: The target state to search for (decimal representation).
        shots: Number of measurement shots.
        max_iterations: Maximum number of iterations to test.

    Returns:
        Tuple of (best_iteration, best_probability).

    Raises:
        ValueError: If target_state is out of range for the given number of qubits.
    """
    max_state = 2**num_qubits - 1
    if target_state > max_state:
        raise ValueError(
            f"Target state {target_state} is out of range for {num_qubits} "
            f"qubits (max: {max_state})"
        )

    # Create the circuit
    grover_circuit = create_grover_circuit(num_qubits, target_state, shots)

    # Calculate optimal iterations
    optimal_iterations = calculate_optimal_iterations(num_qubits)
    if max_iterations is not None:
        optimal_iterations = min(optimal_iterations, max_iterations)

    print("=== Grover Search Configuration ===")
    print(f"Number of qubits: {num_qubits}")
    print(f"Search space size: {2**num_qubits}")
    print(f"Target state: |{format(target_state, f'0{num_qubits}b')}⟩ (decimal: {target_state})")
    print(f"Optimal iterations: {optimal_iterations}")
    print(f"Random search probability: {1/(2**num_qubits):.6f}")
    print()

    # Execute search with different iteration counts
    results = []
    max_test_iterations = min(optimal_iterations + 3, 10)  # Test up to 10 iterations max

    print("=== Search Results ===")
    for iterations in range(max_test_iterations):
        probabilities = grover_circuit(iterations)
        target_probability = probabilities[target_state]
        results.append(target_probability)
        print(f"Iteration {iterations}: Target probability = {target_probability:.6f}")

    # Find best iteration
    best_iteration = np.argmax(results)
    best_probability = results[best_iteration]

    print(f"\n=== Summary ===")
    print(f"Best iteration: {best_iteration}")
    print(f"Best probability: {best_probability:.6f}")
    print(f"Amplification factor: {best_probability / (1/(2**num_qubits)):.2f}x")

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_test_iterations), results, "bo-", linewidth=2, markersize=8)
    plt.axhline(
        y=1 / (2**num_qubits),
        color="r",
        linestyle="--",
        label="Random search probability",
    )
    plt.axvline(x=optimal_iterations, color="g", linestyle="--", label="Theoretical optimal")
    plt.axvline(x=best_iteration, color="orange", linestyle=":", label="Actual best")
    plt.xlabel("Iteration Count")
    plt.ylabel("Probability of Finding Target")
    plt.title(f'Grover Search: {num_qubits} qubits, target |{format(target_state, f"0{num_qubits}b")}⟩')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Show final probability distribution for best iteration
    if best_iteration > 0:
        final_probabilities = grover_circuit(best_iteration)
        print(f"\n=== Final Probability Distribution (Iteration {best_iteration}) ===")

        # Show top 10 most probable states
        sorted_indices = np.argsort(final_probabilities)[::-1][:10]
        for i, index in enumerate(sorted_indices):
            state_label = f"|{format(index, f'0{num_qubits}b')}⟩"
            is_target = " ← TARGET" if index == target_state else ""
            print(f"{i+1:2d}. {state_label}: {final_probabilities[index]:.6f}{is_target}")

    return best_iteration, best_probability


# ====== Usage Examples ======

if __name__ == "__main__":
    # Example 1: 5 qubits, find state 9 (|01001⟩)
    print("Example 1: 5 qubits, target state 9 (|01001⟩)")
    run_grover_search(num_qubits=5, target_state=9, shots=5000)

    print("\n" + "=" * 60 + "\n")

    # Example 2: 3 qubits, find state 5 (|101⟩)
    print("Example 2: 3 qubits, target state 5 (|101⟩)")
    run_grover_search(num_qubits=3, target_state=5, shots=2000)

    print("\n" + "=" * 60 + "\n")

    # Example 3: 4 qubits, find state 12 (|1100⟩)
    print("Example 3: 4 qubits, target state 12 (|1100⟩)")
    run_grover_search(num_qubits=4, target_state=12, shots=3000)

