#!/usr/bin/env python3
"""
CRISPR Optimizer Runner Script

This script demonstrates the quantum CRISPR optimization algorithm
for guide RNA selection and analysis.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

from src.medical.crispr_optimizer import QuantumCRISPROptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for CRISPR optimization."""
    parser = argparse.ArgumentParser(description='Quantum CRISPR Guide RNA Optimizer')
    parser.add_argument('--target', type=str, required=True,
                       help='Target DNA sequence for CRISPR cutting')
    parser.add_argument('--num-qubits', type=int, default=16,
                       help='Number of qubits for quantum simulation')
    parser.add_argument('--max-iter', type=int, default=100,
                       help='Maximum optimization iterations')
    parser.add_argument('--output', type=str, default='crispr_results.txt',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("🧬 Quantum CRISPR Guide RNA Optimizer")
    print("=" * 50)
    print(f"Target sequence: {args.target}")
    print(f"Qubits: {args.num_qubits}")
    print(f"Max iterations: {args.max_iter}")
    
    # Initialize optimizer
    try:
        optimizer = QuantumCRISPROptimizer(
            target_sequence=args.target,
            num_qubits=args.num_qubits
        )
        
        # Run optimization
        logger.info("Starting CRISPR optimization...")
        result = optimizer.optimize_guides(max_iter=args.max_iter)
        
        if result.get('success'):
            print(f"\n✅ Optimization successful!")
            print(f"Best energy: {result['best_energy']:.6f}")
            print(f"Optimal parameters: {len(result['optimal_parameters'])} values")
            
            # Analyze guides
            guides = optimizer.analyze_guides(result)
            
            print(f"\n📊 Guide RNA Analysis:")
            for i, guide in enumerate(guides[:5]):  # Top 5 guides
                print(f"  Guide {i+1}: {guide['sequence']}")
                print(f"    On-target score: {guide['on_target_score']:.4f}")
                print(f"    Off-target penalty: {guide['off_target_penalty']:.4f}")
                print(f"    Overall score: {guide['overall_score']:.4f}")
            
            # Save results
            with open(args.output, 'w') as f:
                f.write("Quantum CRISPR Optimization Results\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Target sequence: {args.target}\n")
                f.write(f"Best energy: {result['best_energy']:.6f}\n")
                f.write(f"Optimization iterations: {result.get('iterations', 'N/A')}\n\n")
                
                f.write("Top Guide RNAs:\n")
                for i, guide in enumerate(guides):
                    f.write(f"{i+1}. {guide['sequence']} (score: {guide['overall_score']:.4f})\n")
            
            print(f"\n💾 Results saved to: {args.output}")
            
        else:
            print(f"\n❌ Optimization failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"CRISPR optimization failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
