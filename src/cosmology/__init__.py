"""
Cosmology and Astrophysics Quantum Computing Applications

This module contains quantum algorithms for cosmological simulations,
black hole physics, dark matter research, and astrophysical phenomena.
"""

from .black_hole_simulation import QuantumBlackHoleSimulator
from .dark_matter_detection import QuantumDarkMatterDetector
from .gravitational_waves import QuantumGravitationalWaveAnalyzer
from .hawking_radiation import HawkingRadiationSimulator
from .quantum_field_theory import QuantumFieldTheorySimulator

__all__ = [
    'QuantumBlackHoleSimulator',
    'HawkingRadiationSimulator',
    'QuantumDarkMatterDetector',
    'QuantumGravitationalWaveAnalyzer',
    'QuantumFieldTheorySimulator'
]
