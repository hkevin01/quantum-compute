"""
Simple Quantum Computing Examples for Testing and Learning

This module contains basic quantum computing examples to help understand
fundamental concepts and test the quantum computing environment.
"""

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicQuantumExamples:
    """Collection of basic quantum computing examples."""
    
    def __init__(self):
        """Initialize with default simulator."""
        self.simulator = AerSimulator()
        logger.info("Initialized Basic Quantum Examples")
    
    def example_1_single_qubit_superposition(self) -> Dict:
        """
        Example 1: Single qubit in superposition.
        
        Creates a single qubit in superposition using Hadamard gate
        and measures the result multiple times.
        """
        print("\n🔬 Example 1: Single Qubit Superposition")
        print("=" * 50)
        
        # Create quantum circuit with 1 qubit and 1 classical bit
        qc = QuantumCircuit(1, 1)
        
        # Apply Hadamard gate to create superposition
        qc.h(0)
        
        # Measure the qubit
        qc.measure(0, 0)
        
        print("Circuit created:")
        print(qc.draw())
        
        # Execute the circuit
        transpiled_qc = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled_qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nMeasurement results (1000 shots):")
        for state, count in counts.items():
            percentage = (count / 1000) * 100
            print(f"  |{state}⟩: {count} times ({percentage:.1f}%)")
        
        print(f"\n✨ Expected: ~50% |0⟩ and ~50% |1⟩ (due to superposition)")
        
        return {
            'circuit': qc,
            'results': counts,
            'description': 'Single qubit superposition with Hadamard gate'
        }
    
    def example_2_two_qubit_entanglement(self) -> Dict:
        """
        Example 2: Two-qubit entanglement (Bell state).
        
        Creates a maximally entangled Bell state between two qubits.
        """
        print("\n🔬 Example 2: Two-Qubit Entanglement (Bell State)")
        print("=" * 50)
        
        # Create quantum circuit with 2 qubits and 2 classical bits
        qc = QuantumCircuit(2, 2)
        
        # Create Bell state |00⟩ + |11⟩
        qc.h(0)      # Put first qubit in superposition
        qc.cx(0, 1)  # Entangle second qubit with first
        
        # Measure both qubits
        qc.measure([0, 1], [0, 1])
        
        print("Circuit created:")
        print(qc.draw())
        
        # Execute the circuit
        transpiled_qc = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled_qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nMeasurement results (1000 shots):")
        for state, count in counts.items():
            percentage = (count / 1000) * 100
            print(f"  |{state}⟩: {count} times ({percentage:.1f}%)")
        
        print(f"\n✨ Expected: ~50% |00⟩ and ~50% |11⟩ (entangled qubits)")
        print("  Notice: No |01⟩ or |10⟩ states due to entanglement!")
        
        return {
            'circuit': qc,
            'results': counts,
            'description': 'Bell state demonstrating quantum entanglement'
        }
    
    def example_3_quantum_interference(self) -> Dict:
        """
        Example 3: Quantum interference demonstration.
        
        Shows how quantum states can interfere constructively or destructively.
        """
        print("\n🔬 Example 3: Quantum Interference")
        print("=" * 50)
        
        # Create circuit demonstrating interference
        qc = QuantumCircuit(1, 1)
        
        # Create superposition
        qc.h(0)
        
        # Apply phase rotation (this will affect interference)
        qc.rz(np.pi, 0)  # π phase rotation
        
        # Apply another Hadamard (this creates interference)
        qc.h(0)
        
        # Measure
        qc.measure(0, 0)
        
        print("Circuit created:")
        print(qc.draw())
        
        # Execute the circuit
        transpiled_qc = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled_qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nMeasurement results (1000 shots):")
        for state, count in counts.items():
            percentage = (count / 1000) * 100
            print(f"  |{state}⟩: {count} times ({percentage:.1f}%)")
        
        print(f"\n✨ Expected: ~100% |1⟩ due to destructive interference of |0⟩")
        print("  The phase rotation causes the |0⟩ amplitude to cancel out!")
        
        return {
            'circuit': qc,
            'results': counts,
            'description': 'Quantum interference with phase rotation'
        }
    
    def example_4_three_qubit_ghz_state(self) -> Dict:
        """
        Example 4: Three-qubit GHZ state.
        
        Creates a three-qubit maximally entangled state.
        """
        print("\n🔬 Example 4: Three-Qubit GHZ State")
        print("=" * 50)
        
        # Create quantum circuit with 3 qubits and 3 classical bits
        qc = QuantumCircuit(3, 3)
        
        # Create GHZ state |000⟩ + |111⟩
        qc.h(0)        # Put first qubit in superposition
        qc.cx(0, 1)    # Entangle second qubit
        qc.cx(0, 2)    # Entangle third qubit
        
        # Measure all qubits
        qc.measure([0, 1, 2], [0, 1, 2])
        
        print("Circuit created:")
        print(qc.draw())
        
        # Execute the circuit
        transpiled_qc = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled_qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nMeasurement results (1000 shots):")
        for state, count in sorted(counts.items()):
            percentage = (count / 1000) * 100
            print(f"  |{state}⟩: {count} times ({percentage:.1f}%)")
        
        print(f"\n✨ Expected: ~50% |000⟩ and ~50% |111⟩")
        print("  All three qubits are maximally entangled!")
        
        return {
            'circuit': qc,
            'results': counts,
            'description': 'Three-qubit GHZ state showing multi-qubit entanglement'
        }
    
    def example_5_parameterized_circuit(self, theta: float = np.pi/4) -> Dict:
        """
        Example 5: Parameterized quantum circuit.
        
        Demonstrates how to create circuits with variable parameters.
        """
        print(f"\n🔬 Example 5: Parameterized Circuit (θ = {theta:.3f})")
        print("=" * 50)
        
        # Create parameterized circuit
        qc = QuantumCircuit(2, 2)
        
        # Apply rotation gates with parameter theta
        qc.ry(theta, 0)        # Y-rotation on first qubit
        qc.rx(theta/2, 1)      # X-rotation on second qubit
        
        # Create entanglement
        qc.cx(0, 1)
        
        # Apply more parameterized gates
        qc.rz(theta, 1)        # Z-rotation on second qubit
        
        # Measure
        qc.measure([0, 1], [0, 1])
        
        print("Circuit created:")
        print(qc.draw())
        
        # Execute the circuit
        transpiled_qc = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled_qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nMeasurement results (1000 shots):")
        for state, count in sorted(counts.items()):
            percentage = (count / 1000) * 100
            print(f"  |{state}⟩: {count} times ({percentage:.1f}%)")
        
        print(f"\n✨ Results depend on θ parameter")
        print(f"  Try different values to see how the distribution changes!")
        
        return {
            'circuit': qc,
            'results': counts,
            'parameter': theta,
            'description': f'Parameterized circuit with theta = {theta:.3f}'
        }
    
    def example_6_quantum_random_number_generator(self) -> Dict:
        """
        Example 6: Quantum random number generator.
        
        Uses quantum superposition to generate truly random numbers.
        """
        print("\n🔬 Example 6: Quantum Random Number Generator")
        print("=" * 50)
        
        # Create circuit with 4 qubits for 4-bit random number
        num_bits = 4
        qc = QuantumCircuit(num_bits, num_bits)
        
        # Put all qubits in superposition
        for i in range(num_bits):
            qc.h(i)
        
        # Measure all qubits
        qc.measure(range(num_bits), range(num_bits))
        
        print("Circuit created:")
        print(qc.draw())
        
        # Generate multiple random numbers
        random_numbers = []
        for _ in range(10):
            transpiled_qc = transpile(qc, self.simulator)
            job = self.simulator.run(transpiled_qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            # Get the measured bit string and convert to integer
            bit_string = list(counts.keys())[0]
            random_number = int(bit_string, 2)
            random_numbers.append(random_number)
        
        print(f"\n10 quantum random numbers (0-{2**num_bits-1}):")
        print(f"  {random_numbers}")
        
        # Show distribution over many measurements
        transpiled_qc = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled_qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"\nDistribution over 1000 measurements:")
        for state, count in sorted(counts.items(), key=lambda x: int(x[0], 2)):
            decimal = int(state, 2)
            percentage = (count / 1000) * 100
            print(f"  {decimal:2d} (|{state}⟩): {count:3d} times ({percentage:.1f}%)")
        
        print(f"\n✨ Each 4-bit number should appear ~equally often (~6.25% each)")
        
        return {
            'circuit': qc,
            'results': counts,
            'random_numbers': random_numbers,
            'description': f'{num_bits}-bit quantum random number generator'
        }
    
    def run_all_examples(self) -> List[Dict]:
        """Run all basic quantum examples."""
        print("🚀 Running All Basic Quantum Computing Examples")
        print("=" * 60)
        
        examples = []
        
        # Run each example
        examples.append(self.example_1_single_qubit_superposition())
        examples.append(self.example_2_two_qubit_entanglement())
        examples.append(self.example_3_quantum_interference())
        examples.append(self.example_4_three_qubit_ghz_state())
        examples.append(self.example_5_parameterized_circuit())
        examples.append(self.example_6_quantum_random_number_generator())
        
        print("\n🎉 All examples completed!")
        print("\nKey Quantum Concepts Demonstrated:")
        print("  1. Superposition - Qubits existing in multiple states")
        print("  2. Entanglement - Quantum correlations between qubits")
        print("  3. Interference - Quantum amplitudes can cancel or reinforce")
        print("  4. Multi-qubit systems - Complex quantum states")
        print("  5. Parameterized circuits - Variable quantum operations")
        print("  6. Practical applications - Random number generation")
        
        return examples


def demo_basic_quantum_examples():
    """Demonstration function for basic quantum examples."""
    examples = BasicQuantumExamples()
    results = examples.run_all_examples()
    
    # Optional: Save results summary
    print(f"\n📊 Summary:")
    for i, result in enumerate(results, 1):
        print(f"  Example {i}: {result['description']}")
    
    return results


if __name__ == "__main__":
    demo_basic_quantum_examples()
