"""
Quantum Circuit Builder and Analysis Utilities

This module provides utilities for building common quantum circuits,
analyzing entanglement, and performing circuit optimization.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import Statevector, entropy, partial_trace
from qiskit_aer import AerSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumCircuitBuilder:
    """Utility class for building common quantum circuits."""
    
    @staticmethod
    def create_bell_state_circuit(qubit_pairs: List[Tuple[int, int]]) -> QuantumCircuit:
        """Create Bell state between specified qubit pairs."""
        max_qubit = max(max(pair) for pair in qubit_pairs) + 1
        qc = QuantumCircuit(max_qubit)
        
        for q1, q2 in qubit_pairs:
            qc.h(q1)
            qc.cx(q1, q2)
        
        return qc
    
    @staticmethod
    def create_ghz_state_circuit(qubits: List[int]) -> QuantumCircuit:
        """Create GHZ state across specified qubits."""
        max_qubit = max(qubits) + 1
        qc = QuantumCircuit(max_qubit)
        
        qc.h(qubits[0])
        for i in range(1, len(qubits)):
            qc.cx(qubits[0], qubits[i])
        
        return qc
    
    @staticmethod
    def create_quantum_fourier_transform(num_qubits: int) -> QuantumCircuit:
        """Create Quantum Fourier Transform circuit."""
        qc = QuantumCircuit(num_qubits)
        
        for i in range(num_qubits):
            qc.h(i)
            for j in range(i + 1, num_qubits):
                qc.cp(np.pi / 2**(j - i), j, i)
        
        # Reverse qubit order
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - 1 - i)
        
        return qc
    
    @staticmethod
    def create_variational_circuit(num_qubits: int, layers: int = 3, 
                                 entanglement: str = 'linear') -> QuantumCircuit:
        """Create variational quantum circuit for optimization."""
        return RealAmplitudes(
            num_qubits=num_qubits,
            reps=layers,
            entanglement=entanglement,
            insert_barriers=True
        )
    
    @staticmethod
    def create_feature_encoding_circuit(data: List[float], 
                                      encoding: str = 'angle') -> QuantumCircuit:
        """Create circuit for encoding classical data."""
        num_qubits = len(data)
        qc = QuantumCircuit(num_qubits)
        
        if encoding == 'angle':
            for i, value in enumerate(data):
                qc.ry(value, i)
        elif encoding == 'amplitude':
            # Simplified amplitude encoding
            normalized_data = np.array(data)
            normalized_data = normalized_data / np.linalg.norm(normalized_data)
            qc.initialize(normalized_data, range(num_qubits))
        
        return qc


class EntanglementAnalyzer:
    """Utility class for analyzing quantum entanglement."""
    
    def __init__(self, backend=None):
        self.backend = backend or AerSimulator()
    
    def calculate_entanglement_entropy(self, qc: QuantumCircuit, 
                                     subsystem_qubits: List[int]) -> float:
        """Calculate entanglement entropy of subsystem."""
        try:
            # Get statevector
            statevector = Statevector.from_instruction(qc)
            
            # Calculate reduced density matrix
            remaining_qubits = [i for i in range(qc.num_qubits) 
                              if i not in subsystem_qubits]
            
            if remaining_qubits:
                rho_reduced = partial_trace(statevector, remaining_qubits)
                return entropy(rho_reduced, base=2)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Failed to calculate entanglement entropy: {e}")
            return 0.0
    
    def measure_entanglement_depth(self, qc: QuantumCircuit) -> int:
        """Estimate entanglement depth of quantum circuit."""
        # Count layers of two-qubit gates
        two_qubit_layers = 0
        current_layer_qubits = set()
        
        for instruction in qc.data:
            if len(instruction[1]) == 2:  # Two-qubit gate
                q1, q2 = instruction[1][0].index, instruction[1][1].index
                
                if q1 in current_layer_qubits or q2 in current_layer_qubits:
                    # Start new layer
                    two_qubit_layers += 1
                    current_layer_qubits = {q1, q2}
                else:
                    current_layer_qubits.update({q1, q2})
        
        return two_qubit_layers
    
    def analyze_circuit_entanglement(self, qc: QuantumCircuit) -> Dict:
        """Comprehensive entanglement analysis of quantum circuit."""
        analysis = {
            'num_qubits': qc.num_qubits,
            'circuit_depth': qc.depth(),
            'num_two_qubit_gates': 0,
            'entanglement_depth': self.measure_entanglement_depth(qc),
            'connectivity_map': {}
        }
        
        # Count gates and build connectivity
        connectivity = {}
        for instruction in qc.data:
            if len(instruction[1]) == 2:  # Two-qubit gate
                analysis['num_two_qubit_gates'] += 1
                q1, q2 = instruction[1][0].index, instruction[1][1].index
                
                if q1 not in connectivity:
                    connectivity[q1] = set()
                if q2 not in connectivity:
                    connectivity[q2] = set()
                    
                connectivity[q1].add(q2)
                connectivity[q2].add(q1)
        
        analysis['connectivity_map'] = {k: list(v) for k, v in connectivity.items()}
        
        # Calculate potential entanglement (theoretical maximum)
        max_possible_entanglement = min(qc.num_qubits, analysis['num_two_qubit_gates'])
        analysis['entanglement_ratio'] = (analysis['num_two_qubit_gates'] / 
                                        max_possible_entanglement if max_possible_entanglement > 0 else 0)
        
        return analysis
    
    def find_entangled_subsystems(self, qc: QuantumCircuit) -> List[List[int]]:
        """Find connected components (entangled subsystems) in circuit."""
        # Build adjacency graph from two-qubit gates
        adjacency = {i: set() for i in range(qc.num_qubits)}
        
        for instruction in qc.data:
            if len(instruction[1]) == 2:
                q1, q2 = instruction[1][0].index, instruction[1][1].index
                adjacency[q1].add(q2)
                adjacency[q2].add(q1)
        
        # Find connected components using DFS
        visited = set()
        subsystems = []
        
        for qubit in range(qc.num_qubits):
            if qubit not in visited:
                component = []
                stack = [qubit]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(adjacency[current] - visited)
                
                subsystems.append(sorted(component))
        
        return subsystems


def demo_circuit_utilities():
    """Demonstration of quantum circuit utilities."""
    print("🔧 Quantum Circuit Utilities Demonstration")
    print("=" * 50)
    
    # Create various circuit types
    builder = QuantumCircuitBuilder()
    analyzer = EntanglementAnalyzer()
    
    # Bell state
    print("Creating Bell state circuit...")
    bell_circuit = builder.create_bell_state_circuit([(0, 1), (2, 3)])
    bell_analysis = analyzer.analyze_circuit_entanglement(bell_circuit)
    
    print(f"Bell state analysis:")
    print(f"  Circuit depth: {bell_analysis['circuit_depth']}")
    print(f"  Two-qubit gates: {bell_analysis['num_two_qubit_gates']}")
    print(f"  Entanglement depth: {bell_analysis['entanglement_depth']}")
    
    # GHZ state
    print("\nCreating GHZ state circuit...")
    ghz_circuit = builder.create_ghz_state_circuit([0, 1, 2, 3])
    ghz_analysis = analyzer.analyze_circuit_entanglement(ghz_circuit)
    
    print(f"GHZ state analysis:")
    print(f"  Circuit depth: {ghz_analysis['circuit_depth']}")
    print(f"  Two-qubit gates: {ghz_analysis['num_two_qubit_gates']}")
    
    # Variational circuit
    print("\nCreating variational circuit...")
    var_circuit = builder.create_variational_circuit(6, layers=2)
    var_analysis = analyzer.analyze_circuit_entanglement(var_circuit)
    
    print(f"Variational circuit analysis:")
    print(f"  Circuit depth: {var_analysis['circuit_depth']}")
    print(f"  Entanglement ratio: {var_analysis['entanglement_ratio']:.3f}")
    
    # Find entangled subsystems
    subsystems = analyzer.find_entangled_subsystems(var_circuit)
    print(f"  Entangled subsystems: {subsystems}")
    
    print("\n✅ Circuit utilities demonstration completed!")


if __name__ == "__main__":
    demo_circuit_utilities()
