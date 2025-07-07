"""
Unit tests for medical quantum computing modules.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestQuantumCRISPROptimizer(unittest.TestCase):
    """Test cases for QuantumCRISPROptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.target_sequence = "ATGGATTTATCTGCTCTTCGCGTT"
        self.num_qubits = 8
    
    @patch('src.medical.crispr_optimizer.AerSimulator')
    def test_optimizer_initialization(self, mock_simulator):
        """Test CRISPR optimizer initialization."""
        from src.medical.crispr_optimizer import QuantumCRISPROptimizer
        
        optimizer = QuantumCRISPROptimizer(
            target_sequence=self.target_sequence,
            num_qubits=self.num_qubits
        )
        
        self.assertEqual(optimizer.target_sequence, self.target_sequence)
        self.assertEqual(optimizer.num_qubits, self.num_qubits)
        self.assertIsNotNone(optimizer.backend)
    
    def test_guide_scoring(self):
        """Test guide RNA scoring functions."""
        from src.medical.crispr_optimizer import QuantumCRISPROptimizer
        
        optimizer = QuantumCRISPROptimizer(
            target_sequence=self.target_sequence,
            num_qubits=self.num_qubits
        )
        
        # Test on-target scoring
        guide_rna = "GATTTATCTGCTCTTCGCGT"
        score = optimizer.calculate_on_target_score(guide_rna)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestProteinFoldingVQE(unittest.TestCase):
    """Test cases for ProteinFoldingVQE."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.amino_sequence = "ACDEFG"
        self.num_qubits = 6
    
    @patch('src.medical.protein_folding.AerSimulator')
    def test_protein_folder_initialization(self, mock_simulator):
        """Test protein folding VQE initialization."""
        from src.medical.protein_folding import ProteinFoldingVQE
        
        folder = ProteinFoldingVQE(
            amino_acid_sequence=self.amino_sequence,
            num_qubits=self.num_qubits
        )
        
        self.assertEqual(folder.sequence, self.amino_sequence)
        self.assertEqual(folder.num_qubits, self.num_qubits)
        self.assertEqual(folder.num_residues, len(self.amino_sequence))
    
    def test_hamiltonian_creation(self):
        """Test Hamiltonian creation for protein folding."""
        from src.medical.protein_folding import ProteinFoldingVQE
        
        folder = ProteinFoldingVQE(
            amino_acid_sequence=self.amino_sequence,
            num_qubits=self.num_qubits
        )
        
        hamiltonian = folder.create_hamiltonian()
        self.assertIsNotNone(hamiltonian)


class TestQuantumDrugDiscovery(unittest.TestCase):
    """Test cases for QuantumDrugDiscovery."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_qubits = 8
    
    @patch('src.medical.drug_discovery.AerSimulator')
    def test_drug_discovery_initialization(self, mock_simulator):
        """Test drug discovery system initialization."""
        from src.medical.drug_discovery import QuantumDrugDiscovery
        
        discovery = QuantumDrugDiscovery(num_qubits=self.num_qubits)
        
        self.assertEqual(discovery.num_qubits, self.num_qubits)
        self.assertIsNotNone(discovery.fragments)
    
    def test_molecular_hamiltonian_creation(self):
        """Test molecular Hamiltonian creation."""
        from src.medical.drug_discovery import QuantumDrugDiscovery
        
        discovery = QuantumDrugDiscovery(num_qubits=self.num_qubits)
        
        target_properties = {
            'max_weight': 400,
            'target_logp': 2.5,
            'drug_like': True
        }
        
        hamiltonian = discovery.create_molecular_hamiltonian(target_properties)
        self.assertIsNotNone(hamiltonian)


class TestQuantumGenomicAnalyzer(unittest.TestCase):
    """Test cases for QuantumGenomicAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_qubits = 12
        self.test_sequence = "ATGCGTACGT"
    
    @patch('src.medical.genomic_analysis.AerSimulator')
    def test_genomic_analyzer_initialization(self, mock_simulator):
        """Test genomic analyzer initialization."""
        from src.medical.genomic_analysis import QuantumGenomicAnalyzer
        
        analyzer = QuantumGenomicAnalyzer(num_qubits=self.num_qubits)
        
        self.assertEqual(analyzer.num_qubits, self.num_qubits)
        self.assertIsNotNone(analyzer.base_encoding)
    
    def test_sequence_encoding(self):
        """Test DNA sequence encoding and decoding."""
        from src.medical.genomic_analysis import QuantumGenomicAnalyzer
        
        analyzer = QuantumGenomicAnalyzer(num_qubits=self.num_qubits)
        
        # Test encoding
        binary = analyzer.encode_sequence(self.test_sequence)
        self.assertIsInstance(binary, str)
        self.assertEqual(len(binary), len(self.test_sequence) * 2)
        
        # Test decoding
        decoded = analyzer.decode_sequence(binary)
        self.assertEqual(decoded, self.test_sequence)


class TestQuantumBiomarkerDiscovery(unittest.TestCase):
    """Test cases for QuantumBiomarkerDiscovery."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_features = 6
        self.num_qubits = 6
    
    @patch('src.medical.biomarker_discovery.AerSimulator')
    def test_biomarker_discovery_initialization(self, mock_simulator):
        """Test biomarker discovery initialization."""
        from src.medical.biomarker_discovery import QuantumBiomarkerDiscovery
        
        discovery = QuantumBiomarkerDiscovery(
            num_features=self.num_features,
            num_qubits=self.num_qubits
        )
        
        self.assertEqual(discovery.num_features, self.num_features)
        self.assertEqual(discovery.num_qubits, self.num_qubits)
    
    def test_training_data_preparation(self):
        """Test training data preparation."""
        from src.medical.biomarker_discovery import QuantumBiomarkerDiscovery
        
        discovery = QuantumBiomarkerDiscovery(
            num_features=self.num_features,
            num_qubits=self.num_qubits
        )
        
        # Create mock data
        gene_expression = np.random.random((10, self.num_features)).tolist()
        labels = [0, 1] * 5
        
        X, y = discovery.prepare_training_data(gene_expression, labels)
        
        self.assertEqual(X.shape, (10, self.num_features))
        self.assertEqual(y.shape, (10,))
        self.assertTrue(np.all(X >= 0))
        self.assertTrue(np.all(X <= 2 * np.pi))


if __name__ == '__main__':
    unittest.main()
