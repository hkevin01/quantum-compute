"""
Simple Quantum Algorithm Examples

This module contains implementations of famous quantum algorithms
in their simplest forms for educational purposes.
"""

import logging
from typing import Dict, List

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleQuantumAlgorithms:
    """Collection of simple quantum algorithm implementations."""
    
    def __init__(self):
        """Initialize with default simulator."""
        self.simulator = AerSimulator()
        logger.info("Initialized Simple Quantum Algorithms")
    
    def deuteeron_josza_algorithm(self, oracle_type: str = "constant") -> Dict:
        """
        Simplified Deutsch-Josza algorithm demonstration.
        
        Determines if a function is constant or balanced with just one query.
        
        Args:
            oracle_type: "constant" or "balanced"
        """
        print(f"\n🔍 Deutsch-Josza Algorithm ({oracle_type} oracle)")
        print("=" * 50)
        
        # Create circuit with 2 qubits (1 input + 1 ancilla)
        qc = QuantumCircuit(2, 1)
        
        # Initialize ancilla qubit to |1⟩
        qc.x(1)
        
        # Apply Hadamard to both qubits
        qc.h(0)
        qc.h(1)
        
        # Apply oracle function
        if oracle_type == "constant":
            # Constant function: do nothing (f(x) = 0) or flip ancilla (f(x) = 1)
            # For f(x) = 0, we do nothing
            pass
        elif oracle_type == "balanced":
            # Balanced function: flip ancilla if input is |1⟩
            qc.cx(0, 1)
        
        # Apply Hadamard to input qubit
        qc.h(0)
        
        # Measure input qubit
        qc.measure(0, 0)
        
        print("Circuit:")
        print(qc.draw())
        
        # Execute
        job = self.simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nResults: {counts}")
        
        # Interpret results
        if '0' in counts and counts.get('0', 0) > 500:
            conclusion = "Function is CONSTANT"
        else:
            conclusion = "Function is BALANCED"
        
        print(f"Conclusion: {conclusion}")
        print(f"Expected: Function is {oracle_type.upper()}")
        
        return {
            'oracle_type': oracle_type,
            'results': counts,
            'conclusion': conclusion,
            'circuit': qc
        }
    
    def grover_search_2qubits(self, target_state: str = "11") -> Dict:
        """
        Simplified 2-qubit Grover's search algorithm.
        
        Searches for a specific 2-bit target state.
        
        Args:
            target_state: Target state to search for ("00", "01", "10", or "11")
        """
        print(f"\n🔍 Grover's Search Algorithm (target: |{target_state}⟩)")
        print("=" * 50)
        
        qc = QuantumCircuit(2, 2)
        
        # Step 1: Initialize superposition
        qc.h(0)
        qc.h(1)
        print("Step 1: Created superposition of all states")
        
        # Step 2: Oracle - mark the target state
        print(f"Step 2: Marking target state |{target_state}⟩")
        
        # Oracle implementation depends on target
        if target_state == "00":
            # Flip phase for |00⟩
            qc.x(0)
            qc.x(1)
            qc.cz(0, 1)
            qc.x(0)
            qc.x(1)
        elif target_state == "01":
            # Flip phase for |01⟩
            qc.x(0)
            qc.cz(0, 1)
            qc.x(0)
        elif target_state == "10":
            # Flip phase for |10⟩
            qc.x(1)
            qc.cz(0, 1)
            qc.x(1)
        elif target_state == "11":
            # Flip phase for |11⟩
            qc.cz(0, 1)
        
        # Step 3: Diffusion operator (amplitude amplification)
        print("Step 3: Amplifying target amplitude")
        qc.h(0)
        qc.h(1)
        qc.x(0)
        qc.x(1)
        qc.cz(0, 1)
        qc.x(0)
        qc.x(1)
        qc.h(0)
        qc.h(1)
        
        # Measure
        qc.measure([0, 1], [0, 1])
        
        print("Final circuit:")
        print(qc.draw())
        
        # Execute
        job = self.simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nResults: {counts}")
        
        # Find most common result
        most_common = max(counts.items(), key=lambda x: x[1])
        found_state = most_common[0]
        probability = most_common[1] / 1000
        
        print(f"Most frequent result: |{found_state}⟩ ({probability:.1%})")
        print(f"Target was: |{target_state}⟩")
        
        success = found_state == target_state
        print(f"Search {'SUCCESS' if success else 'FAILED'}!")
        
        return {
            'target_state': target_state,
            'found_state': found_state,
            'success': success,
            'probability': probability,
            'results': counts,
            'circuit': qc
        }
    
    def quantum_phase_estimation_simple(self, phase: float = 0.25) -> Dict:
        """
        Simplified quantum phase estimation.
        
        Estimates the phase of a simple Z-rotation gate.
        
        Args:
            phase: Phase to estimate (as fraction of 2π)
        """
        print(f"\n🔍 Quantum Phase Estimation (phase = {phase} × 2π)")
        print("=" * 50)
        
        # Use 2 counting qubits for simplicity
        qc = QuantumCircuit(3, 2)  # 2 counting + 1 eigenstate qubit
        
        # Initialize eigenstate |1⟩ for Z gate
        qc.x(2)
        
        # Initialize counting qubits in superposition
        qc.h(0)
        qc.h(1)
        
        # Controlled phase operations
        # For qubit 0: apply U^1 = Z^(2π × phase)
        qc.cp(2 * np.pi * phase, 0, 2)
        
        # For qubit 1: apply U^2 = Z^(2π × phase × 2)
        qc.cp(2 * np.pi * phase * 2, 1, 2)
        
        # Inverse QFT on counting qubits
        qc.h(1)
        qc.cp(-np.pi/2, 0, 1)
        qc.h(0)
        
        # Measure counting qubits
        qc.measure([0, 1], [0, 1])
        
        print("Circuit:")
        print(qc.draw())
        
        # Execute
        job = self.simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nResults: {counts}")
        
        # Interpret results
        most_common = max(counts.items(), key=lambda x: x[1])
        measured_binary = most_common[0]
        measured_decimal = int(measured_binary, 2)
        
        # Convert to estimated phase
        estimated_phase = measured_decimal / 4  # 2^2 = 4 for 2 qubits
        
        print(f"Measured binary: {measured_binary}")
        print(f"Estimated phase: {estimated_phase} × 2π")
        print(f"Actual phase: {phase} × 2π")
        print(f"Error: {abs(estimated_phase - phase):.3f}")
        
        return {
            'actual_phase': phase,
            'estimated_phase': estimated_phase,
            'error': abs(estimated_phase - phase),
            'results': counts,
            'circuit': qc
        }
    
    def quantum_fourier_transform_demo(self, input_state: str = "101") -> Dict:
        """
        Demonstration of Quantum Fourier Transform.
        
        Args:
            input_state: Binary string representing input state
        """
        print(f"\n🔍 Quantum Fourier Transform (input: |{input_state}⟩)")
        print("=" * 50)
        
        n_qubits = len(input_state)
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize input state
        for i, bit in enumerate(reversed(input_state)):
            if bit == '1':
                qc.x(i)
        
        print(f"Initialized state: |{input_state}⟩")
        
        # Apply QFT
        def qft(circuit, n):
            """Apply QFT to first n qubits."""
            for qubit in range(n):
                circuit.h(qubit)
                for j in range(qubit + 1, n):
                    circuit.cp(np.pi / 2**(j - qubit), j, qubit)
            
            # Swap qubits to reverse order
            for qubit in range(n // 2):
                circuit.swap(qubit, n - qubit - 1)
        
        qft(qc, n_qubits)
        
        # Measure
        qc.measure(range(n_qubits), range(n_qubits))
        
        print("Circuit with QFT:")
        print(qc.draw())
        
        # Execute
        job = self.simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nResults: {counts}")
        print("The QFT spreads the amplitude across all basis states")
        print("with specific phase relationships.")
        
        return {
            'input_state': input_state,
            'results': counts,
            'circuit': qc
        }
    
    def simple_variational_quantum_eigensolver(self) -> Dict:
        """
        Simple VQE example to find ground state of Pauli-Z operator.
        """
        print(f"\n🔍 Simple Variational Quantum Eigensolver")
        print("=" * 50)
        
        # We'll find the ground state of the Z operator
        # Ground state should be |0⟩ with eigenvalue -1
        
        best_energy = float('inf')
        best_angle = 0
        energies = []
        angles = []
        
        print("Optimizing over rotation angles...")
        
        # Simple grid search over angles
        for theta in np.linspace(0, 2*np.pi, 20):
            # Create parameterized circuit
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)  # Parameterized rotation
            
            # Measure expectation value of Z operator
            # For Z operator: <ψ|Z|ψ> = P(0) - P(1)
            measure_qc = qc.copy()
            measure_qc.add_register(ClassicalRegister(1))
            measure_qc.measure(0, 0)
            
            # Execute
            job = self.simulator.run(measure_qc, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            # Calculate expectation value
            prob_0 = counts.get('0', 0) / 1000
            prob_1 = counts.get('1', 0) / 1000
            expectation_value = prob_0 - prob_1  # <Z>
            
            energies.append(expectation_value)
            angles.append(theta)
            
            if expectation_value < best_energy:
                best_energy = expectation_value
                best_angle = theta
        
        print(f"Optimization complete!")
        print(f"Best angle: {best_angle:.3f} radians")
        print(f"Best energy: {best_energy:.3f}")
        print(f"Theoretical minimum: -1.0 at θ = 0")
        print(f"Error: {abs(best_energy - (-1.0)):.3f}")
        
        # Create optimized circuit
        optimal_qc = QuantumCircuit(1, 1)
        optimal_qc.ry(best_angle, 0)
        optimal_qc.measure(0, 0)
        
        return {
            'best_angle': best_angle,
            'best_energy': best_energy,
            'theoretical_energy': -1.0,
            'error': abs(best_energy - (-1.0)),
            'optimization_curve': list(zip(angles, energies)),
            'optimal_circuit': optimal_qc
        }
    
    def run_all_algorithms(self) -> List[Dict]:
        """Run all simple quantum algorithms."""
        print("🚀 Running All Simple Quantum Algorithms")
        print("=" * 60)
        
        results = []
        
        # Run each algorithm
        results.append(self.deuteeron_josza_algorithm("constant"))
        results.append(self.deuteeron_josza_algorithm("balanced"))
        results.append(self.grover_search_2qubits("11"))
        results.append(self.grover_search_2qubits("01"))
        results.append(self.quantum_phase_estimation_simple(0.25))
        results.append(self.quantum_fourier_transform_demo("101"))
        results.append(self.simple_variational_quantum_eigensolver())
        
        print("\n🎉 All algorithms completed!")
        print("\nQuantum Algorithms Demonstrated:")
        print("  1. Deutsch-Josza - Function classification")
        print("  2. Grover's Search - Database search")
        print("  3. Phase Estimation - Eigenvalue estimation")
        print("  4. Quantum Fourier Transform - Frequency analysis")
        print("  5. Variational Quantum Eigensolver - Ground state finding")
        
        return results


def demo_quantum_algorithms():
    """Demonstration of simple quantum algorithms."""
    # Import required for VQE
    from qiskit import ClassicalRegister
    
    algorithms = SimpleQuantumAlgorithms()
    results = algorithms.run_all_algorithms()
    
    print(f"\n📊 Summary of {len(results)} algorithms:")
    for i, result in enumerate(results, 1):
        algorithm_name = list(result.keys())[0] if result else "Unknown"
        print(f"  {i}. {algorithm_name}")
    
    return results


if __name__ == "__main__":
    demo_quantum_algorithms()
