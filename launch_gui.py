#!/usr/bin/env python3
"""
Launcher script for the Quantum Computing Explorer GUI
"""

import os
import sys

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from gui.quantum_explorer import main
    
    if __name__ == "__main__":
        print("🚀 Starting Quantum Computing Explorer GUI...")
        print("📋 Features:")
        print("   • Interactive quantum computing demonstrations")
        print("   • Visual explanations of quantum phenomena")
        print("   • Real quantum circuit simulations")
        print("   • Educational content for all levels")
        print("\n🔧 Requirements:")
        print("   • Python 3.7+")
        print("   • PyQt5: pip install PyQt5")
        print("   • Qiskit: pip install qiskit qiskit-aer")
        print("   • Matplotlib: pip install matplotlib")
        print("\n⚡ Starting GUI...\n")
        
        main()
        
except ImportError as e:
    print(f"❌ Error importing GUI module: {e}")
    print("\n🔧 Installation Help:")
    print("1. Install required packages:")
    print("   pip install PyQt5 qiskit qiskit-aer matplotlib numpy")
    print("\n2. Make sure you're in the project directory")
    print("\n3. Run: python launch_gui.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting GUI: {e}")
    sys.exit(1)
