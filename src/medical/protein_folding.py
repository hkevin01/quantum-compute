"""
Quantum Protein Folding Optimization using VQE

This module implements Variational Quantum Eigensolver (VQE) algorithms
for protein structure prediction and folding energy minimization.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinFoldingVQE:
    """
    Quantum Variational Eigensolver for protein folding optimization.
    
    Uses quantum algorithms to find minimal energy conformations of protein
    structures by encoding amino acid interactions as quantum Hamiltonians.
    """
    
    def __init__(self, amino_acid_sequence: str, num_qubits: int = 8, backend=None):
        """
        Initialize the protein folding VQE optimizer.
        
        Args:
            amino_acid_sequence: Single letter amino acid sequence
            num_qubits: Number of qubits for quantum simulation
            backend: Quantum backend for execution
        """
        self.sequence = amino_acid_sequence.upper()
        self.num_qubits = num_qubits
        self.backend = backend or AerSimulator()
        self.amino_acids = list(self.sequence)
        self.num_residues = len(self.amino_acids)
        
        # Amino acid properties for energy calculations
        self.hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        self.charge = {
            'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1,
            'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
            'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
        }
        
        logger.info(f"Initialized protein folding VQE for sequence: {self.sequence}")
        logger.info(f"Sequence length: {self.num_residues} amino acids")
    
    def create_hamiltonian(self) -> SparsePauliOp:
        """
        Create the Hamiltonian encoding protein folding energy.
        
        Returns:
            SparsePauliOp: Quantum Hamiltonian for the protein system
        """
        pauli_strings = []
        coefficients = []
        
        # Hydrophobic interactions (attractive)
        for i in range(self.num_residues - 1):
            for j in range(i + 1, self.num_residues):
                if i < self.num_qubits and j < self.num_qubits:
                    hydro_i = self.hydrophobicity.get(self.amino_acids[i], 0)
                    hydro_j = self.hydrophobicity.get(self.amino_acids[j], 0)
                    
                    if hydro_i > 0 and hydro_j > 0:  # Both hydrophobic
                        interaction_strength = -0.1 * hydro_i * hydro_j
                        
                        # Create Pauli string for interaction
                        pauli_str = ['I'] * self.num_qubits
                        pauli_str[i] = 'Z'
                        pauli_str[j] = 'Z'
                        
                        pauli_strings.append(''.join(pauli_str))
                        coefficients.append(interaction_strength)
        
        # Electrostatic interactions
        for i in range(self.num_residues - 1):
            for j in range(i + 1, self.num_residues):
                if i < self.num_qubits and j < self.num_qubits:
                    charge_i = self.charge.get(self.amino_acids[i], 0)
                    charge_j = self.charge.get(self.amino_acids[j], 0)
                    
                    if charge_i != 0 and charge_j != 0:
                        # Coulomb-like interaction
                        distance_factor = 1.0 / (abs(j - i) + 1)
                        interaction_strength = 0.2 * charge_i * charge_j * distance_factor
                        
                        # Create Pauli string for electrostatic interaction
                        pauli_str = ['I'] * self.num_qubits
                        pauli_str[i] = 'X'
                        pauli_str[j] = 'X'
                        
                        pauli_strings.append(''.join(pauli_str))
                        coefficients.append(interaction_strength)
        
        # Backbone constraints (keep structure reasonable)
        for i in range(self.num_residues - 1):
            if i < self.num_qubits - 1:
                # Neighboring residues should be correlated
                pauli_str = ['I'] * self.num_qubits
                pauli_str[i] = 'Z'
                pauli_str[i + 1] = 'Z'
                
                pauli_strings.append(''.join(pauli_str))
                coefficients.append(-0.1)  # Slight preference for correlation
        
        # Secondary structure preferences
        for i in range(self.num_residues - 2):
            if i < self.num_qubits - 2:
                # Helix formation preference for certain amino acids
                aa = self.amino_acids[i]
                if aa in ['A', 'E', 'L']:  # Helix-favoring amino acids
                    pauli_str = ['I'] * self.num_qubits
                    pauli_str[i] = 'Y'
                    pauli_str[i + 1] = 'Y'
                    pauli_str[i + 2] = 'Y'
                    
                    pauli_strings.append(''.join(pauli_str))
                    coefficients.append(-0.05)  # Slight helix preference
        
        if not pauli_strings:
            # Fallback: simple identity operator
            pauli_strings = ['I' * self.num_qubits]
            coefficients = [0.0]
        
        hamiltonian = SparsePauliOp(pauli_strings, coeffs=coefficients)
        logger.info(f"Created Hamiltonian with {len(pauli_strings)} terms")
        
        return hamiltonian
    
    def create_ansatz(self, layers: int = 3) -> QuantumCircuit:
        """
        Create variational ansatz circuit for protein folding.
        
        Args:
            layers: Number of layers in the ansatz circuit
            
        Returns:
            QuantumCircuit: Parameterized quantum circuit
        """
        # Use EfficientSU2 ansatz with circular entanglement
        ansatz = EfficientSU2(
            num_qubits=self.num_qubits,
            reps=layers,
            entanglement='circular',
            insert_barriers=True
        )
        
        logger.info(f"Created ansatz with {layers} layers and {ansatz.num_parameters} parameters")
        return ansatz
    
    def optimize_folding(self, max_iter: int = 100, layers: int = 3) -> Dict:
        """
        Run VQE optimization to find minimal energy protein conformation.
        
        Args:
            max_iter: Maximum optimization iterations
            layers: Number of ansatz layers
            
        Returns:
            Dict: Optimization results including energy and parameters
        """
        logger.info("Starting protein folding optimization...")
        
        # Create Hamiltonian and ansatz
        hamiltonian = self.create_hamiltonian()
        ansatz = self.create_ansatz(layers)
        
        # Set up VQE with SPSA optimizer
        optimizer = SPSA(maxiter=max_iter, learning_rate=0.01, perturbation=0.1)
        
        vqe = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=self.backend
        )
        
        # Run optimization
        try:
            result = vqe.compute_minimum_eigenvalue(hamiltonian)
            
            optimal_energy = result.eigenvalue.real
            optimal_params = result.optimal_parameters
            
            logger.info(f"Optimization completed!")
            logger.info(f"Minimal energy found: {optimal_energy:.6f}")
            logger.info(f"Number of function evaluations: {result.cost_function_evals}")
            
            return {
                'energy': optimal_energy,
                'parameters': optimal_params,
                'eigenstate': result.eigenstate,
                'cost_function_evals': result.cost_function_evals,
                'optimizer_time': result.optimizer_time,
                'sequence': self.sequence,
                'num_qubits': self.num_qubits
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return {
                'energy': None,
                'error': str(e),
                'sequence': self.sequence
            }
    
    def analyze_conformation(self, result: Dict) -> Dict:
        """
        Analyze the optimized protein conformation.
        
        Args:
            result: VQE optimization result
            
        Returns:
            Dict: Analysis of the protein structure
        """
        if result.get('energy') is None:
            return {'error': 'No valid optimization result to analyze'}
        
        analysis = {
            'sequence': self.sequence,
            'length': self.num_residues,
            'binding_energy': result['energy'],
            'stability_score': -result['energy'],  # Lower energy = higher stability
        }
        
        # Analyze hydrophobic clustering
        hydrophobic_residues = [i for i, aa in enumerate(self.amino_acids) 
                              if self.hydrophobicity.get(aa, 0) > 2.0]
        analysis['hydrophobic_core_size'] = len(hydrophobic_residues)
        
        # Analyze charge distribution
        charged_residues = [i for i, aa in enumerate(self.amino_acids) 
                          if self.charge.get(aa, 0) != 0]
        analysis['charged_residues'] = len(charged_residues)
        
        # Secondary structure prediction (simplified)
        helix_favoring = ['A', 'E', 'L', 'M']
        sheet_favoring = ['V', 'I', 'F', 'Y']
        
        helix_content = sum(1 for aa in self.amino_acids if aa in helix_favoring)
        sheet_content = sum(1 for aa in self.amino_acids if aa in sheet_favoring)
        
        analysis['predicted_helix_content'] = helix_content / self.num_residues
        analysis['predicted_sheet_content'] = sheet_content / self.num_residues
        
        logger.info(f"Conformation analysis completed for {self.sequence}")
        return analysis
    
    def fold_protein(self, max_iter: int = 100, layers: int = 3) -> Dict:
        """
        Complete protein folding workflow.
        
        Args:
            max_iter: Maximum optimization iterations
            layers: Number of ansatz layers
            
        Returns:
            Dict: Complete folding results and analysis
        """
        # Run optimization
        result = self.optimize_folding(max_iter, layers)
        
        # Analyze results
        if result.get('energy') is not None:
            analysis = self.analyze_conformation(result)
            result['analysis'] = analysis
        
        return result


def demo_protein_folding():
    """Demonstration of protein folding VQE algorithm."""
    print("🧬 Quantum Protein Folding Demonstration")
    print("=" * 50)
    
    # Test with a small peptide sequence
    test_sequence = "ACDEFGHIK"  # 9 amino acids
    
    print(f"Folding protein sequence: {test_sequence}")
    print(f"Sequence length: {len(test_sequence)} amino acids")
    
    # Initialize folder
    folder = ProteinFoldingVQE(test_sequence, num_qubits=8)
    
    # Perform folding
    result = folder.fold_protein(max_iter=50, layers=2)
    
    if result.get('energy') is not None:
        print(f"\n✅ Folding completed successfully!")
        print(f"Optimal energy: {result['energy']:.6f}")
        print(f"Function evaluations: {result['cost_function_evals']}")
        
        if 'analysis' in result:
            analysis = result['analysis']
            print(f"\n📊 Structure Analysis:")
            print(f"Stability score: {analysis['stability_score']:.6f}")
            print(f"Hydrophobic core size: {analysis['hydrophobic_core_size']}")
            print(f"Charged residues: {analysis['charged_residues']}")
            print(f"Predicted helix content: {analysis['predicted_helix_content']:.2%}")
            print(f"Predicted sheet content: {analysis['predicted_sheet_content']:.2%}")
    else:
        print(f"\n❌ Folding failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    demo_protein_folding()
