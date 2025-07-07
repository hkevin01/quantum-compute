#!/usr/bin/env python3
"""
Quantum Computing Explorer GUI

An interactive GUI for exploring quantum computing examples and algorithms.
Each tab demonstrates a different quantum concept with visualizations and explanations.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PyQt5 not found. Install with: pip install PyQt5")
    sys.exit(1)

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

try:
    import basic_quantum_examples
    import quantum_algorithms
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit.quantum_info import Statevector
    from qiskit.visualization import plot_bloch_multivector, plot_histogram
    from qiskit_aer import AerSimulator
except ImportError as e:
    print(f"Warning: Could not import quantum modules: {e}")
    print("Make sure Qiskit is installed: pip install qiskit qiskit-aer")
    print("Make sure Qiskit is installed: pip install qiskit qiskit-aer")



class QuantumExplorerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Computing Explorer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize simulator
        self.simulator = AerSimulator()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_introduction_tab()
        self.create_superposition_tab()
        self.create_entanglement_tab()
        self.create_interference_tab()
        self.create_quantum_algorithms_tab()
        self.create_quantum_ml_tab()
        self.create_medical_applications_tab()
        self.create_cosmology_tab()
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #007acc;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9f;
            }
            QPushButton:pressed {
                background-color: #004080;
            }
        """)
    
    def create_introduction_tab(self):
        """Introduction to quantum computing concepts"""
        widget = QWidget()
        self.tab_widget.addTab(widget, "🚀 Introduction")
        
        layout = QVBoxLayout(widget)
        
        # Create scrollable text area
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Arial", 11))
        
        content = """
🌟 Welcome to Quantum Computing Explorer! 🌟

This interactive application demonstrates the fascinating world of quantum computing through practical examples and visualizations.

🔬 What Makes Quantum Computing Special?

Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously. This leads to three key quantum phenomena:

1. 🌊 SUPERPOSITION
   • Qubits can be in a combination of 0 and 1 states simultaneously
   • Like a coin spinning in the air - it's both heads and tails until it lands
   • Classical equivalent: Checking every path in a maze one by one
   • Quantum advantage: Exploring all paths simultaneously

2. 🔗 ENTANGLEMENT
   • Qubits can be mysteriously connected across any distance
   • Measuring one instantly affects its entangled partner
   • Einstein called this "spooky action at a distance"
   • Enables quantum teleportation and cryptography

3. 🌀 INTERFERENCE
   • Quantum states can amplify or cancel each other out
   • Like waves in water - they can constructively or destructively interfere
   • Allows quantum algorithms to amplify correct answers and cancel wrong ones

🚀 Why This Matters for Research:

🧬 MEDICAL APPLICATIONS:
   • Drug Discovery: Simulate molecular interactions with exponential speedup
   • Protein Folding: Model complex protein structures that classical computers struggle with
   • CRISPR Optimization: Find optimal gene editing targets much faster

🌌 COSMOLOGY & PHYSICS:
   • Black Hole Simulations: Model quantum effects near event horizons
   • Dark Matter Detection: Optimize detector configurations for rare events
   • Quantum Field Theory: Simulate fundamental particle interactions

🤖 QUANTUM MACHINE LEARNING:
   • Pattern Recognition: Quantum neural networks for complex data
   • Optimization: Solve NP-hard problems more efficiently
   • Financial Modeling: Risk analysis with quantum Monte Carlo methods

📊 Current Reality:
   • We're in the NISQ era (Noisy Intermediate-Scale Quantum)
   • Quantum computers have 50-1000 qubits with high error rates
   • Quantum advantage exists for specific problems, not general computing
   • Hybrid classical-quantum algorithms show the most promise

🎯 Explore the Tabs:
   • Each tab demonstrates a specific quantum concept
   • Run examples to see quantum behavior in action
   • Compare classical vs quantum approaches
   • Understand why quantum computers excel at certain problems

Ready to explore the quantum realm? Click through the tabs to see quantum computing in action! 🚀
        """
        
        text_edit.setHtml(content.replace('\n', '<br>'))
        layout.addWidget(text_edit)
    
    def create_superposition_tab(self):
        """Superposition demonstration"""
        widget = QWidget()
        self.tab_widget.addTab(widget, "🌊 Superposition")
        
        layout = QHBoxLayout(widget)
        
        # Create splitter for description and demo
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Description panel
        desc_widget = QWidget()
        desc_layout = QVBoxLayout(desc_widget)
        
        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setFont(QFont("Arial", 10))
        desc_text.setMaximumWidth(400)
        
        superposition_desc = """
🌊 QUANTUM SUPERPOSITION

What is it?
Superposition allows a qubit to exist in multiple states simultaneously until measured. It's like a coin spinning in the air - both heads and tails at once.

🔬 The Science:
• A qubit can be |0⟩, |1⟩, or any combination: α|0⟩ + β|1⟩
• |α|² + |β|² = 1 (probabilities must sum to 1)
• When measured, the qubit "collapses" to either 0 or 1

🚀 Why It's Powerful:
• Classical bit: Can check one solution at a time
• Quantum qubit: Can explore multiple solutions simultaneously
• n qubits = 2ⁿ possible states explored at once!

🎯 Real-World Applications:

🧬 Drug Discovery:
• Classical: Test each molecular configuration individually
• Quantum: Explore all configurations in superposition
• Result: Exponentially faster drug screening

🔐 Cryptography:
• Quantum key distribution uses superposition
• Any eavesdropping collapses the quantum state
• Provides mathematically proven security

🎲 Random Number Generation:
• Classical: Pseudo-random (deterministic algorithms)
• Quantum: True randomness from measurement collapse
• Critical for cryptography and simulations

📊 The Demo:
Click "Create Superposition" to see a qubit in equal superposition of |0⟩ and |1⟩. The visualization shows the qubit's state. When you measure it multiple times, you'll see the probabilistic nature of quantum mechanics!
        """
        
        desc_text.setPlainText(superposition_desc)
        desc_layout.addWidget(desc_text)
        splitter.addWidget(desc_widget)
        
        # Demo panel
        demo_widget = QWidget()
        demo_layout = QVBoxLayout(demo_widget)
        
        # Controls
        control_layout = QHBoxLayout()
        
        create_btn = QPushButton("Create Superposition")
        create_btn.clicked.connect(self.demo_superposition)
        create_btn.setStyleSheet("background-color: lightblue; color: black;")
        control_layout.addWidget(create_btn)
        
        measure_btn = QPushButton("Measure 1000 Times")
        measure_btn.clicked.connect(self.measure_superposition)
        measure_btn.setStyleSheet("background-color: lightgreen; color: black;")
        control_layout.addWidget(measure_btn)
        
        control_layout.addStretch()
        demo_layout.addLayout(control_layout)
        
        # Results display
        self.superposition_results = QTextEdit()
        self.superposition_results.setFont(QFont("Courier", 10))
        self.superposition_results.setMaximumHeight(200)
        demo_layout.addWidget(self.superposition_results)
        
        # Matplotlib figure
        self.superposition_figure = Figure(figsize=(12, 6))
        self.superposition_canvas = FigureCanvas(self.superposition_figure)
        demo_layout.addWidget(self.superposition_canvas)
        
        splitter.addWidget(demo_widget)
        splitter.setSizes([400, 800])
    
    def demo_superposition(self):
        """Demonstrate quantum superposition"""
        try:
            # Create a qubit in superposition
            qc = QuantumCircuit(1, 1)
            qc.h(0)  # Hadamard gate creates superposition
            
            # Get the statevector
            statevector = Statevector.from_instruction(qc)
            
            # Clear previous plots
            self.superposition_figure.clear()
            ax1 = self.superposition_figure.add_subplot(1, 2, 1)
            ax2 = self.superposition_figure.add_subplot(1, 2, 2)
            
            # Plot circuit
            qc.draw(output='mpl', ax=ax1)
            ax1.set_title("Quantum Circuit: H|0⟩ = (|0⟩ + |1⟩)/√2")
            
            # Plot probabilities
            probs = statevector.probabilities()
            ax2.bar(['|0⟩', '|1⟩'], probs, color=['blue', 'red'])
            ax2.set_title("State Probabilities")
            ax2.set_ylabel("Probability")
            ax2.set_ylim(0, 1)
            
            self.superposition_canvas.draw()
            
            self.superposition_results.clear()
            self.superposition_results.append(
                f"✅ Superposition Created!\n"
                f"State: {statevector}\n"
                f"Probabilities: |0⟩: 50%, |1⟩: 50%\n"
                f"The qubit is now in equal superposition!\n\n"
                f"🔬 What's happening:\n"
                f"• The Hadamard gate (H) puts the qubit in superposition\n"
                f"• Before measurement, it's both |0⟩ AND |1⟩\n"
                f"• Each measurement has 50% chance of either outcome\n"
                f"• Click 'Measure 1000 Times' to see the statistics!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create superposition: {e}")
    
    def measure_superposition(self):
        """Measure the superposition many times"""
        try:
            # Create and measure the circuit many times
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            
            # Run many shots
            compiled_circuit = transpile(qc, self.simulator)
            job = self.simulator.run(compiled_circuit, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            # Update results
            self.superposition_results.append(
                f"\n📊 Measurement Results (1000 shots):\n")
            for outcome, count in counts.items():
                percentage = (count/1000) * 100
                self.superposition_results.append(
                    f"  |{outcome}⟩: {count} times ({percentage:.1f}%)\n")
            
            self.superposition_results.append(
                f"\n🎯 Notice: Results are close to 50/50!\n"
                f"This proves the qubit was in true superposition.\n"
                f"Classical randomness would give the same statistics,\n"
                f"but quantum superposition explores BOTH states simultaneously!\n")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not measure superposition: {e}")
    
    def create_entanglement_tab(self):
        """Quantum entanglement demonstration"""
        widget = QWidget()
        self.tab_widget.addTab(widget, "🔗 Entanglement")
        
        layout = QVBoxLayout(widget)
        label = QLabel("Entanglement demonstration coming soon...")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 16))
        layout.addWidget(label)
    
    def create_interference_tab(self):
        """Quantum interference demonstration"""
        widget = QWidget()
        self.tab_widget.addTab(widget, "🌀 Interference")
        
        layout = QVBoxLayout(widget)
        label = QLabel("Interference demonstration coming soon...")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 16))
        layout.addWidget(label)
    
    def create_quantum_algorithms_tab(self):
        """Quantum algorithms overview"""
        widget = QWidget()
        self.tab_widget.addTab(widget, "🧮 Algorithms")
        
        layout = QVBoxLayout(widget)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Arial", 11))
        
        algorithms_content = """
🧮 QUANTUM ALGORITHMS SHOWCASE

This tab demonstrates the most important quantum algorithms and their real-world applications.

🔍 GROVER'S SEARCH ALGORITHM
• Problem: Search unsorted database of N items
• Classical: O(N) time - must check every item
• Quantum: O(√N) time - quadratic speedup!
• Applications:
  🔐 Breaking symmetric cryptography
  🧬 Database search in bioinformatics
  🎯 Optimization problems
  💰 Portfolio optimization

⚗️ QUANTUM SIMULATION ALGORITHMS
• Problem: Simulate quantum systems (molecules, materials)
• Classical: Exponentially hard (2^n states)
• Quantum: Natural fit - quantum simulates quantum!
• Applications:
  💊 Drug discovery and design
  🔋 Battery materials optimization
  🧪 Catalyst development
  ⚡ Superconductor research

🔢 SHOR'S FACTORING ALGORITHM
• Problem: Factor large integers
• Classical: Exponential time (breaks down for large numbers)
• Quantum: Polynomial time - exponential speedup!
• Impact:
  🔐 Breaks RSA encryption
  🏛️ Threatens current cybersecurity
  🛡️ Motivates quantum-resistant cryptography

🎯 VARIATIONAL QUANTUM ALGORITHMS (VQE, QAOA)
• Hybrid classical-quantum approach
• Good for NISQ devices (current quantum computers)
• Applications:
  🧬 Protein folding optimization
  🚗 Route optimization
  📈 Financial portfolio optimization
  🔋 Energy system optimization

🤖 QUANTUM MACHINE LEARNING
• Quantum Neural Networks
• Quantum Support Vector Machines
• Quantum Feature Maps
• Applications:
  🏥 Medical diagnosis
  🔬 Drug discovery
  📊 Pattern recognition
  🎯 Optimization

📊 CURRENT STATUS:
• Most algorithms show theoretical advantage
• Practical advantage limited by current hardware
• NISQ algorithms (VQE, QAOA) most promising near-term
• Fault-tolerant quantum computers needed for full advantage

🚀 FUTURE OUTLOOK:
As quantum computers scale up:
• Exponential advantages will become practical
• New algorithms continue to be discovered
• Hybrid approaches bridge classical and quantum
• Industry applications will transform multiple fields

Each algorithm leverages quantum phenomena (superposition, entanglement, interference) in unique ways to achieve computational advantages impossible classically.
        """
        
        text_edit.setPlainText(algorithms_content)
        layout.addWidget(text_edit)
    
    def create_quantum_ml_tab(self):
        """Quantum machine learning applications"""
        widget = QWidget()
        self.tab_widget.addTab(widget, "🤖 Quantum ML")
        
        layout = QVBoxLayout(widget)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Arial", 11))
        
        ml_content = """
🤖 QUANTUM MACHINE LEARNING

Quantum computing promises to revolutionize machine learning through quantum-enhanced algorithms that could provide exponential speedups for certain problems.

[Content continues with detailed explanations of quantum ML concepts...]
        """
        
        text_edit.setPlainText(ml_content)
        layout.addWidget(text_edit)
    
    def create_medical_applications_tab(self):
        """Medical and biological applications"""
        widget = QWidget()
        self.tab_widget.addTab(widget, "🧬 Medical Apps")
        
        layout = QVBoxLayout(widget)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Arial", 11))
        
        medical_content = """
🧬 QUANTUM COMPUTING IN MEDICINE

Quantum computing promises to revolutionize healthcare through unprecedented computational power for biological systems.

[Content continues with detailed medical applications...]
        """
        
        text_edit.setPlainText(medical_content)
        layout.addWidget(text_edit)
    
    def create_cosmology_tab(self):
        """Cosmology and physics applications"""
        widget = QWidget()
        self.tab_widget.addTab(widget, "🌌 Cosmology")
        
        layout = QVBoxLayout(widget)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Arial", 11))
        
        cosmo_content = """
🌌 QUANTUM COMPUTING IN COSMOLOGY

Quantum computing opens new frontiers in understanding the universe, from black holes to dark matter to the fundamental nature of reality itself.

[Content continues with detailed cosmology applications...]
        """
        
        text_edit.setPlainText(cosmo_content)
        layout.addWidget(text_edit)

def main():
    """Main function to run the Quantum Explorer GUI"""
    app = QApplication(sys.argv)
    app.setApplicationName("Quantum Computing Explorer")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = QuantumExplorerGUI()
    window.show()
    
    # Handle closing gracefully
    def cleanup():
        plt.close('all')  # Close all matplotlib figures
    
    app.aboutToQuit.connect(cleanup)
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
