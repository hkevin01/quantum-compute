# Qiskit and Quantum Computing Guide

This guide provides an overview of Qiskit framework and quantum computing concepts as used in this research project.

## 📚 Table of Contents

1. [Introduction to Quantum Computing](#introduction-to-quantum-computing)
2. [Qiskit Framework Overview](#qiskit-framework-overview)
3. [Quantum Algorithms in This Project](#quantum-algorithms-in-this-project)
4. [Implementation Examples](#implementation-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Further Reading](#further-reading)

## 🌟 Introduction to Quantum Computing

### What is Quantum Computing?

Quantum computing leverages quantum mechanical phenomena like **superposition**, **entanglement**, and **interference** to process information in fundamentally different ways than classical computers.

#### Key Quantum Concepts

- **Qubit**: The basic unit of quantum information, can exist in superposition of |0⟩ and |1⟩ states
- **Superposition**: A qubit can be in a combination of both 0 and 1 states simultaneously
- **Entanglement**: Quantum correlation between qubits that persists regardless of distance
- **Interference**: Quantum states can interfere constructively or destructively
- **Measurement**: Forces a quantum system to collapse to a classical state

### Quantum Advantage

Quantum computers can potentially solve certain problems exponentially faster than classical computers:

- **Factoring large numbers** (Shor's algorithm)
- **Searching unsorted databases** (Grover's algorithm)
- **Simulating quantum systems** (Quantum simulation)
- **Optimization problems** (QAOA, VQE)

## 🚀 Qiskit Framework Overview

[Qiskit](https://qiskit.org/) is IBM's open-source quantum computing framework for Python. It provides tools for creating, manipulating, and executing quantum circuits.

### Core Qiskit Components

#### 1. Qiskit Terra (Foundation)
```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator
```

**Purpose**: Core quantum circuit construction and execution

#### 2. Qiskit Aer (Simulators)
```python
from qiskit_aer import AerSimulator, StatevectorSimulator, UnitarySimulator
```

**Purpose**: High-performance quantum circuit simulators

#### 3. Qiskit Nature (Chemistry & Physics)
```python
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
```

**Purpose**: Quantum algorithms for natural sciences

#### 4. Qiskit Machine Learning
```python
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms import VQC
```

**Purpose**: Quantum machine learning algorithms

#### 5. Qiskit Optimization
```python
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
```

**Purpose**: Quantum optimization algorithms

### Basic Quantum Circuit Creation

```python
# Create a quantum circuit with 3 qubits and 3 classical bits
qc = QuantumCircuit(3, 3)

# Add quantum gates
qc.h(0)  # Hadamard gate (creates superposition)
qc.cx(0, 1)  # CNOT gate (creates entanglement)
qc.measure_all()  # Measure all qubits

# Execute on simulator
simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1000).result()
counts = result.get_counts()
```

## 🧬 Quantum Algorithms in This Project

### 1. QAOA (Quantum Approximate Optimization Algorithm)

Used in [`QuantumCRISPROptimizer`](../src/medical/crispr_optimizer.py) for guide RNA optimization:

```python
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.quantum_info import SparsePauliOp

# Create cost Hamiltonian for CRISPR optimization
def create_cost_hamiltonian(self, potential_guides, genome_sites):
    pauli_strings = []
    coefficients = []
    
    for i, guide in enumerate(potential_guides):
        # Encode optimization objective
        on_target_score = self.calculate_on_target_score(guide)
        off_target_penalty = self.calculate_off_target_penalty(guide, genome_sites)
        
        # Create Pauli operator
        pauli_str = 'Z' * self.num_qubits
        coefficient = -(on_target_score - off_target_penalty)
        
        pauli_strings.append(pauli_str)
        coefficients.append(coefficient)
    
    return SparsePauliOp(pauli_strings, coeffs=coefficients)
```

**Applications in this project**:
- CRISPR guide RNA optimization
- Protein folding energy minimization
- Drug molecule optimization

### 2. VQE (Variational Quantum Eigensolver)

For finding ground state energies of molecular systems:

```python
from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal

# Create ansatz circuit
ansatz = TwoLocal(num_qubits, 'ry', 'cz', reps=3, entanglement='linear')

# Set up VQE
vqe = VQE(ansatz, optimizer=SPSA(maxiter=100))
result = vqe.compute_minimum_eigenvalue(hamiltonian)
```

**Applications**:
- Molecular energy calculations
- Protein structure prediction
- Chemical reaction pathway optimization

### 3. Quantum Machine Learning

Using quantum neural networks for classification:

```python
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms import VQC

# Create quantum neural network
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
ansatz = RealAmplitudes(num_qubits, reps=3)

qnn = CircuitQNN(
    circuit=feature_map.compose(ansatz),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

# Use in variational quantum classifier
vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=SPSA())
```

**Applications**:
- Medical diagnosis from genomic data
- Drug-target interaction prediction
- Biomarker discovery

## 💻 Implementation Examples

### Example 1: Basic Quantum Circuit for Superposition

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

def create_superposition_circuit(num_qubits):
    """Create a circuit that puts qubits in superposition."""
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Apply Hadamard gates to create superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc

# Execute and visualize
qc = create_superposition_circuit(3)
simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1000).result()
counts = result.get_counts()

print("Measurement results:", counts)
```

### Example 2: Quantum Entanglement

```python
def create_bell_state():
    """Create a Bell state (maximally entangled state)."""
    qc = QuantumCircuit(2, 2)
    
    # Create Bell state |00⟩ + |11⟩
    qc.h(0)  # Put first qubit in superposition
    qc.cx(0, 1)  # Entangle with second qubit
    
    qc.measure_all()
    return qc

# Test entanglement
bell_circuit = create_bell_state()
result = simulator.run(transpile(bell_circuit, simulator), shots=1000).result()
counts = result.get_counts()
print("Bell state measurements:", counts)
# Expected: roughly 50% |00⟩ and 50% |11⟩
```

### Example 3: Parameterized Quantum Circuit

```python
from qiskit.circuit import Parameter

def create_parameterized_circuit():
    """Create a parameterized circuit for optimization."""
    # Define parameters
    theta = Parameter('θ')
    phi = Parameter('φ')
    
    qc = QuantumCircuit(2)
    
    # Parameterized gates
    qc.ry(theta, 0)
    qc.rz(phi, 1)
    qc.cx(0, 1)
    qc.ry(theta * 2, 1)
    
    return qc

# Bind parameters and execute
param_circuit = create_parameterized_circuit()
bound_circuit = param_circuit.bind_parameters({
    param_circuit.parameters[0]: 0.5,  # theta
    param_circuit.parameters[1]: 1.2   # phi
})
```

## 🔧 Best Practices

### 1. Circuit Design

```python
# ✅ Good: Minimize circuit depth
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

# ❌ Avoid: Unnecessary gates
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.h(1)  # Unnecessary if followed by measurement
qc.h(1)  # Cancels previous H gate
```

### 2. Error Mitigation

```python
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import complete_meas_cal

# Create noise model for realistic simulation
noise_model = NoiseModel.from_backend(backend)

# Measurement error mitigation
meas_calibs, state_labels = complete_meas_cal(qr=qc.qregs[0], circs=qc)
```

### 3. Optimization

```python
# Use appropriate optimization levels
from qiskit import transpile

# For NISQ devices
transpiled_qc = transpile(
    qc, 
    backend=backend,
    optimization_level=3,  # Maximum optimization
    coupling_map=backend.configuration().coupling_map
)
```

### 4. Resource Management

```python
# Efficient simulator usage
simulator = AerSimulator(method='statevector')  # For small circuits
simulator = AerSimulator(method='matrix_product_state')  # For larger circuits

# Batch execution
job = simulator.run(circuits, shots=1000)
results = job.result()
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Circuit Too Deep
```python
# Problem: Circuit depth exceeds device limits
print(f"Circuit depth: {qc.depth()}")

# Solution: Use circuit optimization
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation

pm = PassManager([Optimize1qGates(), CXCancellation()])
optimized_qc = pm.run(qc)
```

#### 2. Memory Issues with Simulation
```python
# Problem: Statevector simulator runs out of memory
# Solution: Use different simulation methods

# For large circuits (>20 qubits)
simulator = AerSimulator(method='matrix_product_state')

# For noisy simulation
simulator = AerSimulator(method='density_matrix')
```

#### 3. Parameter Binding Errors
```python
# Problem: Parameter mismatch in parameterized circuits
# Solution: Check parameter names and types

print("Circuit parameters:", qc.parameters)
param_dict = {param: value for param, value in zip(qc.parameters, values)}
bound_qc = qc.bind_parameters(param_dict)
```

#### 4. Backend Connectivity Issues
```python
# Problem: Circuit not compatible with backend topology
# Solution: Use transpilation with coupling map

from qiskit.transpiler import CouplingMap

coupling_map = CouplingMap.from_line(num_qubits)
transpiled_qc = transpile(qc, coupling_map=coupling_map)
```

## 🔗 Integration with Project Components

### Using with CRISPR Optimizer

```python
from src.medical.crispr_optimizer import QuantumCRISPROptimizer

# Initialize with custom backend
optimizer = QuantumCRISPROptimizer(
    target_sequence="ATGGATTTATCTGCTCTTCGCGTT",
    num_qubits=16
)

# Set custom quantum backend
from qiskit.providers.fake_provider import FakeVigo
optimizer.backend = FakeVigo()
```

### Custom Quantum Algorithms

```python
# Template for new quantum algorithms in this project
class QuantumBioAlgorithm:
    def __init__(self, num_qubits: int, backend=None):
        self.num_qubits = num_qubits
        self.backend = backend or AerSimulator()
        
    def create_circuit(self) -> QuantumCircuit:
        """Override this method for specific algorithms."""
        qc = QuantumCircuit(self.num_qubits)
        # Add algorithm-specific gates
        return qc
        
    def execute(self, shots: int = 1000):
        """Execute the quantum circuit."""
        qc = self.create_circuit()
        transpiled_qc = transpile(qc, self.backend)
        job = self.backend.run(transpiled_qc, shots=shots)
        return job.result()
```

## 📚 Further Reading

### Official Documentation
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Qiskit Textbook](https://qiskit.org/textbook/)
- [Qiskit Tutorials](https://qiskit.org/documentation/tutorials.html)

### Research Papers
- Nielsen & Chuang: "Quantum Computation and Quantum Information"
- Preskill: "Quantum Computing in the NISQ era and beyond"
- Cerezo et al.: "Variational Quantum Algorithms"

### Quantum Algorithms for Biology
- "Quantum algorithms for scientific computing" - Nature Reviews Materials
- "Quantum machine learning for biology and medicine" - Nature
- "Quantum computing for molecular biology" - Current Opinion in Structural Biology

### Project-Specific Resources
- [QAOA Tutorial](https://qiskit.org/textbook/ch-applications/qaoa.html)
- [VQE Implementation Guide](https://qiskit.org/textbook/ch-applications/vqe-molecules.html)
- [Quantum Machine Learning](https://qiskit.org/textbook/ch-machine-learning/)

---

## 🎯 Quick Reference

### Essential Imports
```python
# Core Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Algorithms
from qiskit.algorithms import QAOA, VQE
from qiskit.algorithms.optimizers import SPSA, COBYLA

# Quantum Info
from qiskit.quantum_info import SparsePauliOp, Statevector

# Circuit Library
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
```

### Common Gates
```python
qc.h(0)        # Hadamard (superposition)
qc.x(0)        # Pauli-X (bit flip)
qc.z(0)        # Pauli-Z (phase flip)
qc.cx(0, 1)    # CNOT (entanglement)
qc.ry(θ, 0)    # Y-rotation (parameterized)
qc.measure_all() # Measurement
```

### Simulation Backends
```python
AerSimulator()                           # General purpose
AerSimulator(method='statevector')       # Exact simulation
AerSimulator(method='matrix_product_state') # Large circuits
StatevectorSimulator()                   # State vector only
UnitarySimulator()                       # Unitary matrices
```

This guide serves as a comprehensive reference for quantum computing concepts and Qiskit implementation within this research project. For project-specific implementations, refer to the individual algorithm files in the [`src/`](../src/) directory.
