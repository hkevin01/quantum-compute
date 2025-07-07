#!/usr/bin/env python3
"""
Black Hole Simulation Runner Script

This script demonstrates the quantum black hole simulation
and information paradox exploration.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging

from src.cosmology.black_hole_simulation import QuantumBlackHoleSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for black hole simulation."""
    parser = argparse.ArgumentParser(description='Quantum Black Hole Simulator')
    parser.add_argument('--num-qubits', type=int, default=12,
                       help='Number of qubits for simulation')
    parser.add_argument('--time-steps', type=int, default=10,
                       help='Number of time steps for Hawking radiation')
    parser.add_argument('--mass', type=float, default=1.0,
                       help='Black hole mass (normalized units)')
    parser.add_argument('--output', type=str, default='blackhole_results.json',
                       help='Output file for results')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['formation', 'hawking', 'paradox', 'full'],
                       help='Simulation mode')
    
    args = parser.parse_args()
    
    print("🌌 Quantum Black Hole Simulation")
    print("=" * 50)
    print(f"Qubits: {args.num_qubits}")
    print(f"Black hole mass: {args.mass}")
    print(f"Time steps: {args.time_steps}")
    print(f"Mode: {args.mode}")
    
    # Initialize simulator
    try:
        simulator = QuantumBlackHoleSimulator(num_qubits=args.num_qubits)
        simulator.black_hole_mass = args.mass
        simulator._calculate_black_hole_properties()
        
        results = {}
        
        if args.mode in ['formation', 'full']:
            print("\n🌟 Simulating black hole formation...")
            formation_circuit = simulator.create_black_hole_formation_circuit()
            results['formation'] = {
                'circuit_depth': formation_circuit.depth(),
                'num_gates': len(formation_circuit.data)
            }
            print(f"Formation circuit depth: {results['formation']['circuit_depth']}")
        
        if args.mode in ['hawking', 'full']:
            print("\n🔥 Simulating Hawking radiation...")
            hawking_results = simulator.simulate_hawking_radiation_emission(
                time_steps=args.time_steps
            )
            results['hawking_radiation'] = hawking_results
            
            if hawking_results['radiation_data']:
                final_data = hawking_results['radiation_data'][-1]
                print(f"Final entropy: {final_data['entropy']:.4f}")
                print(f"Final mass: {final_data['black_hole_mass']:.4f}")
        
        if args.mode in ['paradox', 'full']:
            print("\n🔍 Exploring information paradox...")
            paradox_results = simulator.explore_information_paradox()
            results['information_paradox'] = paradox_results
            
            print(f"Information preserved: {paradox_results['information_preserved']}")
            print(f"Paradox resolved: {paradox_results['paradox_resolved']}")
        
        if args.mode == 'full':
            print("\n🚀 Running complete lifecycle simulation...")
            lifecycle = simulator.simulate_complete_black_hole_lifecycle()
            results['complete_lifecycle'] = lifecycle
        
        # Add black hole properties
        results['black_hole_properties'] = {
            'mass': simulator.black_hole_mass,
            'schwarzschild_radius': simulator.event_horizon_radius,
            'hawking_temperature': simulator.hawking_temperature
        }
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {args.output}")
        print("✅ Black hole simulation completed!")
        
    except Exception as e:
        logger.error(f"Black hole simulation failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
