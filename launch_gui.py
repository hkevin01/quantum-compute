#!/usr/bin/env python3
"""
Launcher script for the Quantum Hardware Explorer GUI
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
        print("   • Run shallow circuits on real IBM Quantum hardware")
        print("   • View histograms and circuit diagrams")
        print("\n🔧 Requirements:")
        print("   • Python 3.10+")
        print("   • PyQt5: pip install PyQt5")
        print(
            "   • Qiskit + IBM Runtime: pip install qiskit "
            "qiskit-ibm-runtime"
        )
        print("   • Matplotlib: pip install matplotlib pylatexenc")
        print(
            "   • Env vars set: QISKIT_IBM_TOKEN, QISKIT_IBM_CHANNEL, "
            "QISKIT_IBM_INSTANCE"
        )
        print("\n⚡ Starting GUI...\n")
        
        main()
        
except ImportError as e:
    print(f"❌ Error importing GUI module: {e}")
    print("\n🔧 Installation Help:")
    print("1. Install required packages:")
    print(
        "   pip install PyQt5 qiskit qiskit-ibm-runtime "
        "matplotlib pylatexenc"
    )
    print("\n2. Make sure you're in the project directory")
    print("\n3. Run: python launch_gui.py")
    sys.exit(1)
except RuntimeError as e:
    print(f"❌ Error starting GUI: {e}")
    sys.exit(1)
