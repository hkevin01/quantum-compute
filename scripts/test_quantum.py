#!/usr/bin/env python3
"""
Quick Quantum Test Runner

This script provides a simple way to test quantum computing functionality
and verify that the quantum environment is working correctly.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from examples.basic_quantum_examples import BasicQuantumExamples


def test_quantum_environment():
    """Test that quantum computing environment is working."""
    print("🧪 Testing Quantum Computing Environment")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("Testing imports...")
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        print("  ✅ All imports successful")
        
        # Test basic circuit creation
        print("Testing circuit creation...")
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        print("  ✅ Circuit creation successful")
        
        # Test simulator
        print("Testing quantum simulator...")
        simulator = AerSimulator()
        job = simulator.run(qc, shots=100)
        result = job.result()
        counts = result.get_counts()
        print("  ✅ Quantum simulation successful")
        print(f"  Sample results: {counts}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {str(e)}")
        return False


def run_quick_demo():
    """Run a quick demonstration of quantum computing."""
    print("\n🚀 Quick Quantum Computing Demo")
    print("=" * 50)
    
    examples = BasicQuantumExamples()
    
    # Run just the first few examples for quick testing
    print("Running superposition example...")
    result1 = examples.example_1_single_qubit_superposition()
    
    print("\nRunning entanglement example...")
    result2 = examples.example_2_two_qubit_entanglement()
    
    return [result1, result2]


def run_comprehensive_demo():
    """Run all quantum computing examples."""
    print("\n🔬 Comprehensive Quantum Computing Demo")
    print("=" * 50)
    
    examples = BasicQuantumExamples()
    return examples.run_all_examples()


def interactive_quantum_playground():
    """Interactive quantum circuit playground."""
    print("\n🎮 Interactive Quantum Playground")
    print("=" * 50)
    print("Let's create custom quantum circuits!")
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        while True:
            print("\nOptions:")
            print("1. Create Bell state")
            print("2. Create random superposition")
            print("3. Create custom GHZ state")
            print("4. Exit playground")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                # Bell state
                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure_all()
                
                print("\nCreated Bell state circuit:")
                print(qc.draw())
                
                simulator = AerSimulator()
                result = simulator.run(qc, shots=1000).result()
                counts = result.get_counts()
                print(f"Results: {counts}")
                
            elif choice == '2':
                # Random superposition
                num_qubits = int(input("How many qubits? (1-5): "))
                num_qubits = max(1, min(5, num_qubits))
                
                qc = QuantumCircuit(num_qubits, num_qubits)
                for i in range(num_qubits):
                    qc.h(i)
                qc.measure_all()
                
                print(f"\nCreated {num_qubits}-qubit superposition:")
                print(qc.draw())
                
                simulator = AerSimulator()
                result = simulator.run(qc, shots=1000).result()
                counts = result.get_counts()
                print(f"Results (showing top 5): {dict(list(counts.items())[:5])}")
                
            elif choice == '3':
                # GHZ state
                num_qubits = int(input("How many qubits for GHZ state? (2-4): "))
                num_qubits = max(2, min(4, num_qubits))
                
                qc = QuantumCircuit(num_qubits, num_qubits)
                qc.h(0)
                for i in range(1, num_qubits):
                    qc.cx(0, i)
                qc.measure_all()
                
                print(f"\nCreated {num_qubits}-qubit GHZ state:")
                print(qc.draw())
                
                simulator = AerSimulator()
                result = simulator.run(qc, shots=1000).result()
                counts = result.get_counts()
                print(f"Results: {counts}")
                
            elif choice == '4':
                print("Exiting playground. Thanks for exploring quantum computing!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    except KeyboardInterrupt:
        print("\n\nPlayground interrupted. Goodbye!")
    except Exception as e:
        print(f"Error in playground: {e}")


def benchmark_quantum_performance():
    """Benchmark quantum simulation performance."""
    print("\n⚡ Quantum Performance Benchmark")
    print("=" * 50)
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        simulator = AerSimulator()
        
        # Test different circuit sizes
        for num_qubits in [2, 4, 6, 8, 10]:
            print(f"\nTesting {num_qubits} qubits...")
            
            # Create test circuit
            qc = QuantumCircuit(num_qubits, num_qubits)
            
            # Add some gates
            for i in range(num_qubits):
                qc.h(i)
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            qc.measure_all()
            
            # Time the execution
            start_time = time.time()
            result = simulator.run(qc, shots=1000).result()
            execution_time = time.time() - start_time
            
            print(f"  Circuit depth: {qc.depth()}")
            print(f"  Execution time: {execution_time:.3f} seconds")
            print(f"  States measured: {len(result.get_counts())}")
            
            if execution_time > 5.0:  # Stop if taking too long
                print("  Stopping benchmark (circuits getting too large)")
                break
                
    except Exception as e:
        print(f"Benchmark failed: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Quantum Computing Test Runner')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['test', 'quick', 'full', 'interactive', 'benchmark'],
                       help='Mode to run')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🌟 Quantum Computing Test Runner")
    print("=" * 60)
    
    if args.mode == 'test':
        success = test_quantum_environment()
        if success:
            print("\n🎉 Quantum environment is working correctly!")
            return 0
        else:
            print("\n❌ Quantum environment has issues. Check your installation.")
            return 1
            
    elif args.mode == 'quick':
        if test_quantum_environment():
            run_quick_demo()
        else:
            return 1
            
    elif args.mode == 'full':
        if test_quantum_environment():
            run_comprehensive_demo()
        else:
            return 1
            
    elif args.mode == 'interactive':
        if test_quantum_environment():
            interactive_quantum_playground()
        else:
            return 1
            
    elif args.mode == 'benchmark':
        if test_quantum_environment():
            benchmark_quantum_performance()
        else:
            return 1
    
    print("\n✅ Test runner completed!")
    return 0


if __name__ == "__main__":
    exit(main())
