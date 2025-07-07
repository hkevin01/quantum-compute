#!/usr/bin/env python3
"""
Interactive Quantum Demonstrations

Simple command-line demonstrations of quantum computing concepts
for testing and educational purposes.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit.visualization import plot_histogram
    from qiskit_aer import AerSimulator
except ImportError as e:
    print(f"Warning: Could not import quantum modules: {e}")
    print("Install with: pip install qiskit qiskit-aer matplotlib")


def demo_quantum_coin_flip():
    """Demonstrate quantum vs classical randomness"""
    print("\n🪙 QUANTUM COIN FLIP DEMONSTRATION")
    print("=" * 50)
    
    # Classical coin flip simulation
    print("\n📊 Classical Coin Flip (deterministic):")
    classical_results = []
    for i in range(10):
        result = np.random.choice([0, 1])
        classical_results.append(result)
        print(f"  Flip {i+1}: {'Heads' if result == 0 else 'Tails'}")
    
    # Quantum coin flip
    print("\n⚛️ Quantum Coin Flip (true randomness):")
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Put qubit in superposition
    qc.measure(0, 0)
    
    simulator = AerSimulator()
    quantum_results = []
    
    for i in range(10):
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Get the single measurement result
        outcome = int(list(counts.keys())[0])
        quantum_results.append(outcome)
        print(f"  Flip {i+1}: {'Heads' if outcome == 0 else 'Tails'}")
    
    print(f"\n📈 Statistics:")
    print(f"  Classical: {classical_results.count(0)} heads, {classical_results.count(1)} tails")
    print(f"  Quantum:   {quantum_results.count(0)} heads, {quantum_results.count(1)} tails")
    print(f"\n🎯 Key Difference:")
    print(f"  • Classical: Pseudo-random (deterministic algorithm)")
    print(f"  • Quantum: True randomness from quantum measurement")


def demo_superposition_measurement():
    """Show the probabilistic nature of quantum measurement"""
    print("\n🌊 SUPERPOSITION MEASUREMENT DEMO")
    print("=" * 50)
    
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Create superposition
    
    # Show the quantum state before measurement
    statevector = Statevector.from_instruction(qc)
    print(f"\n🔬 Quantum State: {statevector}")
    print(f"📊 Probabilities: |0⟩: 50%, |1⟩: 50%")
    
    # Measure many times
    qc.measure(0, 0)
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    
    shots_list = [10, 100, 1000, 10000]
    
    for shots in shots_list:
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        zeros = counts.get('0', 0)
        ones = counts.get('1', 0)
        
        print(f"\n📈 Results with {shots:5d} measurements:")
        print(f"  |0⟩: {zeros:5d} times ({zeros/shots*100:5.1f}%)")
        print(f"  |1⟩: {ones:5d} times ({ones/shots*100:5.1f}%)")
    
    print(f"\n🎯 Observation:")
    print(f"  As we increase measurements, results approach 50/50")
    print(f"  This confirms the quantum superposition!")


def demo_bell_state_correlation():
    """Demonstrate quantum entanglement correlations"""
    print("\n🔗 BELL STATE ENTANGLEMENT DEMO")
    print("=" * 50)
    
    # Create Bell state
    qc = QuantumCircuit(2, 2)
    qc.h(0)      # Superposition
    qc.cx(0, 1)  # Entanglement
    qc.measure_all()
    
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print(f"\n🔬 Bell State: |00⟩ + |11⟩")
    print(f"📊 Measurement Results (1000 shots):")
    
    total_same = 0
    total_different = 0
    
    for outcome, count in sorted(counts.items()):
        percentage = (count/1000) * 100
        print(f"  |{outcome}⟩: {count:3d} times ({percentage:5.1f}%)")
        
        if outcome in ['00', '11']:
            total_same += count
        else:
            total_different += count
    
    print(f"\n🎯 Correlation Analysis:")
    print(f"  Same results (00 or 11): {total_same:3d} ({total_same/10:5.1f}%)")
    print(f"  Different results (01 or 10): {total_different:3d} ({total_different/10:5.1f}%)")
    
    if total_different < 50:
        print(f"\n✅ ENTANGLEMENT CONFIRMED!")
        print(f"  The qubits are perfectly correlated!")
        print(f"  This proves quantum entanglement.")
    else:
        print(f"\n⚠️ Results suggest some decoherence in simulation")


def demo_interference_pattern():
    """Show quantum interference in Mach-Zehnder-like setup"""
    print("\n🌀 QUANTUM INTERFERENCE DEMO")
    print("=" * 50)
    
    print("\n🔬 Simulating quantum interference with phase shifts...")
    
    phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    for phase in phases:
        # Create interference circuit
        qc = QuantumCircuit(1, 1)
        qc.h(0)           # First beam splitter
        qc.p(phase, 0)    # Phase shift
        qc.h(0)           # Second beam splitter
        qc.measure(0, 0)
        
        # Measure
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        prob_0 = counts.get('0', 0) / 1000
        prob_1 = counts.get('1', 0) / 1000
        
        print(f"  Phase {phase/np.pi:4.2f}π: |0⟩: {prob_0:5.3f}, |1⟩: {prob_1:5.3f}")
    
    print(f"\n🎯 Observations:")
    print(f"  • Phase 0: Constructive interference → mostly |0⟩")
    print(f"  • Phase π: Destructive interference → mostly |1⟩")
    print(f"  • This demonstrates wave-like quantum behavior!")


def demo_quantum_parallelism():
    """Show quantum parallelism with multiple qubits"""
    print("\n⚡ QUANTUM PARALLELISM DEMO")
    print("=" * 50)
    
    for n_qubits in [1, 2, 3, 4]:
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Put all qubits in superposition
        for i in range(n_qubits):
            qc.h(i)
        
        # Measure all
        qc.measure_all()
        
        # Count possible states
        possible_states = 2 ** n_qubits
        
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        observed_states = len(counts)
        
        print(f"\n📊 {n_qubits} qubits:")
        print(f"  Possible states: {possible_states}")
        print(f"  Observed states: {observed_states}")
        print(f"  States explored simultaneously in superposition!")
        
        # Show some results
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top measurements: {sorted_counts[:3]}")
    
    print(f"\n🚀 Quantum Advantage:")
    print(f"  • Classical: Check each state one by one")
    print(f"  • Quantum: Explore all states simultaneously!")
    print(f"  • With n qubits, quantum explores 2^n states at once")


def interactive_menu():
    """Interactive menu for quantum demonstrations"""
    while True:
        print("\n" + "="*60)
        print("🚀 QUANTUM COMPUTING DEMONSTRATIONS")
        print("="*60)
        print("1. 🪙 Quantum vs Classical Coin Flip")
        print("2. 🌊 Superposition Measurement Statistics")
        print("3. 🔗 Bell State Entanglement Correlations")
        print("4. 🌀 Quantum Interference Patterns")
        print("5. ⚡ Quantum Parallelism Demo")
        print("6. 🎯 Run All Demonstrations")
        print("0. 🚪 Exit")
        print()
        
        try:
            choice = input("Select a demonstration (0-6): ").strip()
            
            if choice == '0':
                print("\n👋 Thanks for exploring quantum computing!")
                break
            elif choice == '1':
                demo_quantum_coin_flip()
            elif choice == '2':
                demo_superposition_measurement()
            elif choice == '3':
                demo_bell_state_correlation()
            elif choice == '4':
                demo_interference_pattern()
            elif choice == '5':
                demo_quantum_parallelism()
            elif choice == '6':
                print("\n🎯 Running all demonstrations...")
                demo_quantum_coin_flip()
                demo_superposition_measurement()
                demo_bell_state_correlation()
                demo_interference_pattern()
                demo_quantum_parallelism()
                print("\n✅ All demonstrations completed!")
            else:
                print("❌ Invalid choice. Please select 0-6.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    try:
        print("🔧 Checking quantum computing environment...")
        
        # Test imports
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        print("✅ Qiskit successfully imported!")
        print("✅ Environment ready for quantum demonstrations!")
        
        interactive_menu()
        
    except ImportError:
        print("❌ Qiskit not found!")
        print("\n🔧 Installation Instructions:")
        print("1. Install Qiskit: pip install qiskit qiskit-aer")
        print("2. Install matplotlib: pip install matplotlib")
        print("3. Run this script again")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
