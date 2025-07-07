"""
Quantum Biomarker Discovery using Machine Learning

This module implements quantum machine learning algorithms for
biomarker discovery, disease classification, and medical diagnosis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import CircuitQNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumBiomarkerDiscovery:
    """
    Quantum machine learning for biomarker discovery and medical diagnosis.
    
    Uses quantum neural networks and variational quantum classifiers
    to identify disease biomarkers from genomic and clinical data.
    """
    
    def __init__(self, num_features: int = 8, num_qubits: int = 8, backend=None):
        """Initialize quantum biomarker discovery system."""
        self.num_features = num_features
        self.num_qubits = num_qubits
        self.backend = backend or AerSimulator()
        self.vqc_model = None
        self.feature_map = None
        self.ansatz = None
        
        logger.info(f"Initialized quantum biomarker discovery system")
        logger.info(f"Features: {num_features}, Qubits: {num_qubits}")
    
    def create_feature_map(self, reps: int = 2) -> ZZFeatureMap:
        """Create quantum feature map for data encoding."""
        feature_map = ZZFeatureMap(
            feature_dimension=self.num_features,
            reps=reps,
            entanglement='linear'
        )
        self.feature_map = feature_map
        logger.info(f"Created feature map with {reps} repetitions")
        return feature_map
    
    def create_ansatz(self, layers: int = 3) -> RealAmplitudes:
        """Create variational ansatz for quantum classifier."""
        ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=layers,
            entanglement='linear'
        )
        self.ansatz = ansatz
        logger.info(f"Created ansatz with {layers} layers")
        return ansatz
    
    def prepare_training_data(self, gene_expression: List[List[float]], 
                            labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and normalize training data."""
        X = np.array(gene_expression)
        y = np.array(labels)
        
        # Normalize features to [0, 2π] for quantum encoding
        X_normalized = np.zeros_like(X)
        for i in range(X.shape[1]):
            feature_min = X[:, i].min()
            feature_max = X[:, i].max()
            if feature_max > feature_min:
                X_normalized[:, i] = 2 * np.pi * (X[:, i] - feature_min) / (feature_max - feature_min)
            else:
                X_normalized[:, i] = np.pi  # Constant feature
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X_normalized, y
    
    def train_biomarker_classifier(self, X: np.ndarray, y: np.ndarray, 
                                 max_iter: int = 100) -> Dict:
        """Train quantum variational classifier for biomarker discovery."""
        logger.info("Training quantum biomarker classifier...")
        
        # Create feature map and ansatz if not exists
        if self.feature_map is None:
            self.create_feature_map()
        if self.ansatz is None:
            self.create_ansatz()
        
        # Create variational quantum classifier
        optimizer = SPSA(maxiter=max_iter, learning_rate=0.01)
        
        self.vqc_model = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=optimizer,
            quantum_instance=self.backend
        )
        
        try:
            # Train the model
            self.vqc_model.fit(X, y)
            
            # Evaluate training accuracy
            train_predictions = self.vqc_model.predict(X)
            train_accuracy = np.mean(train_predictions == y)
            
            logger.info(f"Training completed!")
            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            
            return {
                'model': self.vqc_model,
                'train_accuracy': train_accuracy,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def identify_biomarkers(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str] = None) -> List[Dict]:
        """Identify important biomarkers using feature importance analysis."""
        if self.vqc_model is None:
            logger.error("Model not trained. Train classifier first.")
            return []
        
        logger.info("Analyzing feature importance for biomarker discovery...")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        biomarkers = []
        baseline_accuracy = np.mean(self.vqc_model.predict(X) == y)
        
        # Feature ablation analysis
        for i in range(X.shape[1]):
            # Create modified dataset with feature i set to mean value
            X_modified = X.copy()
            X_modified[:, i] = np.mean(X[:, i])
            
            try:
                modified_predictions = self.vqc_model.predict(X_modified)
                modified_accuracy = np.mean(modified_predictions == y)
                importance = baseline_accuracy - modified_accuracy
                
                biomarkers.append({
                    'feature_name': feature_names[i],
                    'feature_index': i,
                    'importance_score': importance,
                    'baseline_accuracy': baseline_accuracy,
                    'ablated_accuracy': modified_accuracy
                })
                
            except Exception as e:
                logger.warning(f"Failed to analyze feature {i}: {e}")
                biomarkers.append({
                    'feature_name': feature_names[i],
                    'feature_index': i,
                    'importance_score': 0.0,
                    'error': str(e)
                })
        
        # Sort by importance score (descending)
        biomarkers.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        
        logger.info(f"Identified {len(biomarkers)} potential biomarkers")
        return biomarkers
    
    def predict_disease_risk(self, patient_data: List[float]) -> Dict:
        """Predict disease risk for a patient using trained classifier."""
        if self.vqc_model is None:
            return {'error': 'Model not trained'}
        
        # Normalize patient data (simplified - should use training statistics)
        patient_array = np.array(patient_data).reshape(1, -1)
        
        try:
            prediction = self.vqc_model.predict(patient_array)[0]
            prediction_proba = self.vqc_model.predict_proba(patient_array)[0]
            
            return {
                'predicted_class': int(prediction),
                'risk_probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.5,
                'confidence': float(np.max(prediction_proba)),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def generate_biomarker_report(self, biomarkers: List[Dict], 
                                top_n: int = 5) -> str:
        """Generate a comprehensive biomarker discovery report."""
        report = "🧬 Quantum Biomarker Discovery Report\n"
        report += "=" * 50 + "\n\n"
        
        if not biomarkers:
            report += "No biomarkers identified.\n"
            return report
        
        report += f"Top {min(top_n, len(biomarkers))} Biomarkers:\n\n"
        
        for i, marker in enumerate(biomarkers[:top_n]):
            report += f"{i+1}. {marker['feature_name']}\n"
            report += f"   Importance Score: {marker['importance_score']:.4f}\n"
            report += f"   Baseline Accuracy: {marker['baseline_accuracy']:.4f}\n"
            report += f"   Ablated Accuracy: {marker['ablated_accuracy']:.4f}\n"
            
            # Interpret importance
            if marker['importance_score'] > 0.1:
                significance = "High"
            elif marker['importance_score'] > 0.05:
                significance = "Medium"
            else:
                significance = "Low"
            
            report += f"   Clinical Significance: {significance}\n\n"
        
        # Summary statistics
        avg_importance = np.mean([m['importance_score'] for m in biomarkers])
        max_importance = np.max([m['importance_score'] for m in biomarkers])
        
        report += f"Summary Statistics:\n"
        report += f"- Average importance score: {avg_importance:.4f}\n"
        report += f"- Maximum importance score: {max_importance:.4f}\n"
        report += f"- Total biomarkers analyzed: {len(biomarkers)}\n"
        
        return report


def demo_biomarker_discovery():
    """Demonstration of quantum biomarker discovery."""
    print("🔬 Quantum Biomarker Discovery Demonstration")
    print("=" * 50)
    
    # Initialize system
    biomarker_system = QuantumBiomarkerDiscovery(num_features=6, num_qubits=6)
    
    # Simulate gene expression data
    np.random.seed(42)
    
    # Healthy samples (lower expression of disease genes)
    healthy_samples = np.random.normal(0.3, 0.1, (20, 6))
    healthy_labels = [0] * 20
    
    # Disease samples (higher expression of disease genes)
    disease_samples = np.random.normal(0.7, 0.1, (20, 6))
    disease_labels = [1] * 20
    
    # Combine datasets
    gene_expression = healthy_samples.tolist() + disease_samples.tolist()
    labels = healthy_labels + disease_labels
    
    feature_names = [
        "BRCA1", "BRCA2", "TP53", "EGFR", "MYC", "RAS"
    ]
    
    print(f"Training on {len(gene_expression)} samples...")
    
    # Prepare and train
    X, y = biomarker_system.prepare_training_data(gene_expression, labels)
    training_result = biomarker_system.train_biomarker_classifier(X, y, max_iter=50)
    
    if training_result.get('success'):
        print(f"✅ Training successful!")
        print(f"Training accuracy: {training_result['train_accuracy']:.4f}")
        
        # Identify biomarkers
        biomarkers = biomarker_system.identify_biomarkers(X, y, feature_names)
        
        # Generate report
        report = biomarker_system.generate_biomarker_report(biomarkers, top_n=3)
        print("\n" + report)
        
        # Test prediction on new patient
        new_patient = [0.8, 0.6, 0.9, 0.4, 0.7, 0.5]  # High-risk profile
        prediction = biomarker_system.predict_disease_risk(new_patient)
        
        if prediction.get('success'):
            print(f"New Patient Risk Assessment:")
            print(f"  Predicted class: {prediction['predicted_class']}")
            print(f"  Risk probability: {prediction['risk_probability']:.4f}")
            print(f"  Confidence: {prediction['confidence']:.4f}")
        
    else:
        print(f"❌ Training failed: {training_result.get('error')}")


if __name__ == "__main__":
    demo_biomarker_discovery()
