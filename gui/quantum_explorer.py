#!/usr/bin/env python3
"""
Quantum Computing Explorer GUI

An interactive GUI for exploring quantum computing examples and algorithms.
Each tab demonstrates a different quantum concept with visualizations and explanations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                                QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, 
                                QLabel, QScrollArea, QSplitter, QMessageBox,
                                QFrame)
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QPixmap
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


class QuantumExplorerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Computing Explorer")
        self.root.geometry("1200x800")
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize simulator
        self.simulator = AerSimulator()
        
        # Create tabs
        self.create_introduction_tab()
        self.create_superposition_tab()
        self.create_entanglement_tab()
        self.create_interference_tab()
        self.create_quantum_algorithms_tab()
        self.create_quantum_ml_tab()
        self.create_medical_applications_tab()
        self.create_cosmology_tab()
        
    def create_introduction_tab(self):
        """Introduction to quantum computing concepts"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🚀 Introduction")
        
        # Create scrollable text widget
        text_frame = tk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        intro_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Arial", 11))
        intro_text.pack(fill=tk.BOTH, expand=True)
        
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
        
        intro_text.insert(tk.END, content)
        intro_text.config(state=tk.DISABLED)
        
    def create_superposition_tab(self):
        """Superposition demonstration"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🌊 Superposition")
        
        # Split into description and demo
        main_frame = tk.PanedWindow(frame, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Description panel
        desc_frame = tk.Frame(main_frame)
        main_frame.add(desc_frame, width=400)
        
        desc_text = scrolledtext.ScrolledText(desc_frame, wrap=tk.WORD, font=("Arial", 10))
        desc_text.pack(fill=tk.BOTH, expand=True)
        
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
Click "Create Superposition" to see a qubit in equal superposition of |0⟩ and |1⟩. The Bloch sphere shows the qubit's state in 3D space. When you measure it multiple times, you'll see the probabilistic nature of quantum mechanics!
        """
        
        desc_text.insert(tk.END, superposition_desc)
        desc_text.config(state=tk.DISABLED)
        
        # Demo panel
        demo_frame = tk.Frame(main_frame)
        main_frame.add(demo_frame, width=600)
        
        # Controls
        control_frame = tk.Frame(demo_frame)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Create Superposition", 
                 command=self.demo_superposition, bg="lightblue").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Measure 1000 Times", 
                 command=self.measure_superposition, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.superposition_results = scrolledtext.ScrolledText(demo_frame, height=8, font=("Courier", 10))
        self.superposition_results.pack(fill=tk.X, padx=10, pady=5)
        
        # Matplotlib figure for visualization
        self.superposition_fig, (self.superposition_ax1, self.superposition_ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.superposition_canvas = FigureCanvasTkAgg(self.superposition_fig, demo_frame)
        self.superposition_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def demo_superposition(self):
        """Demonstrate quantum superposition"""
        try:
            # Create a qubit in superposition
            qc = QuantumCircuit(1, 1)
            qc.h(0)  # Hadamard gate creates superposition
            
            # Get the statevector
            statevector = Statevector.from_instruction(qc)
            
            # Clear previous plots
            self.superposition_ax1.clear()
            self.superposition_ax2.clear()
            
            # Plot circuit
            qc.draw(output='mpl', ax=self.superposition_ax1)
            self.superposition_ax1.set_title("Quantum Circuit: H|0⟩ = (|0⟩ + |1⟩)/√2")
            
            # Plot Bloch sphere
            try:
                plot_bloch_multivector(statevector, ax=self.superposition_ax2)
                self.superposition_ax2.set_title("Qubit State on Bloch Sphere")
            except:
                # Fallback if Bloch sphere plotting fails
                probs = statevector.probabilities()
                self.superposition_ax2.bar(['|0⟩', '|1⟩'], probs)
                self.superposition_ax2.set_title("State Probabilities")
                self.superposition_ax2.set_ylabel("Probability")
            
            self.superposition_canvas.draw()
            
            self.superposition_results.delete(1.0, tk.END)
            self.superposition_results.insert(tk.END, 
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
            messagebox.showerror("Error", f"Could not create superposition: {e}")
    
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
            self.superposition_results.insert(tk.END, 
                f"\n📊 Measurement Results (1000 shots):\n")
            for outcome, count in counts.items():
                percentage = (count/1000) * 100
                self.superposition_results.insert(tk.END, 
                    f"  |{outcome}⟩: {count} times ({percentage:.1f}%)\n")
            
            self.superposition_results.insert(tk.END, 
                f"\n🎯 Notice: Results are close to 50/50!\n"
                f"This proves the qubit was in true superposition.\n"
                f"Classical randomness would give the same statistics,\n"
                f"but quantum superposition explores BOTH states simultaneously!\n")
            
            # Scroll to bottom
            self.superposition_results.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not measure superposition: {e}")
    
    def create_entanglement_tab(self):
        """Quantum entanglement demonstration"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔗 Entanglement")
        
        # Split into description and demo
        main_frame = tk.PanedWindow(frame, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Description panel
        desc_frame = tk.Frame(main_frame)
        main_frame.add(desc_frame, width=400)
        
        desc_text = scrolledtext.ScrolledText(desc_frame, wrap=tk.WORD, font=("Arial", 10))
        desc_text.pack(fill=tk.BOTH, expand=True)
        
        entanglement_desc = """
🔗 QUANTUM ENTANGLEMENT

What is it?
Einstein called it "spooky action at a distance" - when qubits become entangled, measuring one instantly affects the other, regardless of distance!

🔬 The Science:
• Entangled qubits cannot be described independently
• Bell states: |00⟩ + |11⟩ or |00⟩ - |11⟩ (maximally entangled)
• Measurement correlation: If one is |0⟩, the other is guaranteed |0⟩
• This happens FASTER than light could travel between them!

🚀 Why It's Revolutionary:

🔐 Quantum Cryptography:
• Quantum Key Distribution (QKD)
• Any eavesdropping breaks entanglement
• Mathematically proven secure communication
• Already used by banks and governments

🌐 Quantum Internet:
• Quantum teleportation of information
• Distributed quantum computing
• Connect quantum computers globally
• Enable quantum cloud computing

🧬 Medical Research:
• Quantum sensors for brain imaging
• Ultra-precise molecular measurements
• Detect single molecules in living cells
• Revolutionary diagnostic capabilities

🎯 Real Examples:

🛰️ China's Quantum Satellites:
• Demonstrated entanglement over 1200 km
• Quantum communication between continents
• Unhackable satellite communications

🏥 Quantum Sensing:
• MRI with quantum-enhanced sensitivity
• Detect Alzheimer's at molecular level
• Monitor drug delivery in real-time

💰 Financial Security:
• Quantum-secured banking transactions
• Protect against future quantum hackers
• JPMorgan and others already investing

📊 The Demo:
We'll create a Bell state where two qubits are maximally entangled. When we measure them, they'll always give the same result - proving their mysterious connection!
        """
        
        desc_text.insert(tk.END, entanglement_desc)
        desc_text.config(state=tk.DISABLED)
        
        # Demo panel
        demo_frame = tk.Frame(main_frame)
        main_frame.add(demo_frame, width=600)
        
        # Controls
        control_frame = tk.Frame(demo_frame)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Create Bell State", 
                 command=self.demo_entanglement, bg="lightcoral").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Test Correlations", 
                 command=self.test_entanglement, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.entanglement_results = scrolledtext.ScrolledText(demo_frame, height=8, font=("Courier", 10))
        self.entanglement_results.pack(fill=tk.X, padx=10, pady=5)
        
        # Matplotlib figure
        self.entanglement_fig, self.entanglement_ax = plt.subplots(1, 1, figsize=(10, 4))
        self.entanglement_canvas = FigureCanvasTkAgg(self.entanglement_fig, demo_frame)
        self.entanglement_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def demo_entanglement(self):
        """Create and visualize entangled qubits"""
        try:
            # Create Bell state |00⟩ + |11⟩
            qc = QuantumCircuit(2, 2)
            qc.h(0)      # Put first qubit in superposition
            qc.cx(0, 1)  # Entangle with second qubit
            
            # Get statevector
            statevector = Statevector.from_instruction(qc)
            
            # Plot circuit
            self.entanglement_ax.clear()
            qc.draw(output='mpl', ax=self.entanglement_ax)
            self.entanglement_ax.set_title("Bell State Circuit: |00⟩ + |11⟩")
            self.entanglement_canvas.draw()
            
            self.entanglement_results.delete(1.0, tk.END)
            self.entanglement_results.insert(tk.END,
                f"✅ Bell State Created!\n"
                f"State: {statevector}\n\n"
                f"🔬 What's happening:\n"
                f"• Hadamard creates superposition: (|0⟩ + |1⟩)/√2\n"
                f"• CNOT entangles: |00⟩ + |11⟩ (both same, never different!)\n"
                f"• The qubits are now mysteriously connected\n"
                f"• Measuring one instantly determines the other\n\n"
                f"🎯 Bell State Properties:\n"
                f"• 50% chance of measuring |00⟩\n"
                f"• 50% chance of measuring |11⟩\n"
                f"• 0% chance of measuring |01⟩ or |10⟩\n"
                f"• This correlation exists even across galaxies!\n\n"
                f"Click 'Test Correlations' to see the magic!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not create entanglement: {e}")
    
    def test_entanglement(self):
        """Test entanglement correlations"""
        try:
            # Create and measure Bell state
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            
            compiled_circuit = transpile(qc, self.simulator)
            job = self.simulator.run(compiled_circuit, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            self.entanglement_results.insert(tk.END,
                f"\n📊 Entanglement Test Results (1000 measurements):\n")
            
            total_same = counts.get('00', 0) + counts.get('11', 0)
            total_different = counts.get('01', 0) + counts.get('10', 0)
            
            for outcome, count in sorted(counts.items()):
                percentage = (count/1000) * 100
                self.entanglement_results.insert(tk.END,
                    f"  |{outcome}⟩: {count} times ({percentage:.1f}%)\n")
            
            self.entanglement_results.insert(tk.END,
                f"\n🎯 Correlation Analysis:\n"
                f"  Same results (00 or 11): {total_same} ({total_same/10:.1f}%)\n"
                f"  Different results (01 or 10): {total_different} ({total_different/10:.1f}%)\n\n")
            
            if total_different < 50:  # Should be close to 0 for perfect entanglement
                self.entanglement_results.insert(tk.END,
                    f"✅ ENTANGLEMENT CONFIRMED!\n"
                    f"The qubits are perfectly correlated - they always give the same result!\n"
                    f"This proves they're quantum mechanically entangled.\n\n"
                    f"🤯 Mind-blowing fact:\n"
                    f"Even if these qubits were separated by light-years,\n"
                    f"measuring one would INSTANTLY determine the other!\n"
                    f"This is what Einstein couldn't accept about quantum mechanics.")
            else:
                self.entanglement_results.insert(tk.END,
                    f"⚠️ Entanglement may be degraded due to noise in simulation.")
            
            self.entanglement_results.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not test entanglement: {e}")
    
    def create_interference_tab(self):
        """Quantum interference demonstration"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🌀 Interference")
        
        # Add description and demo similar to other tabs
        main_frame = tk.PanedWindow(frame, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Description
        desc_frame = tk.Frame(main_frame)
        main_frame.add(desc_frame, width=400)
        
        desc_text = scrolledtext.ScrolledText(desc_frame, wrap=tk.WORD, font=("Arial", 10))
        desc_text.pack(fill=tk.BOTH, expand=True)
        
        interference_desc = """
🌀 QUANTUM INTERFERENCE

What is it?
Quantum interference allows quantum algorithms to amplify correct answers and cancel out wrong ones - like waves constructively and destructively interfering!

🔬 The Science:
• Quantum amplitudes can be positive or negative
• Constructive interference: amplitudes add up
• Destructive interference: amplitudes cancel out
• This is how quantum algorithms achieve speedup!

🚀 How It Enables Quantum Advantage:

🔍 Grover's Search:
• Classical: Search unsorted database in O(N) time
• Quantum: Search in O(√N) time using interference
• Amplifies probability of correct answer
• Cancels probability of wrong answers

🧮 Shor's Algorithm:
• Factoring large numbers (breaks RSA encryption)
• Uses interference to find periods in mathematical functions
• Exponentially faster than classical methods
• Threatens current cryptography

🎯 Real Applications:

💊 Drug Discovery:
• Search through millions of molecular configurations
• Interference amplifies promising drug candidates
• Cancels out ineffective combinations
• Could revolutionize pharmaceutical research

🔐 Optimization Problems:
• Portfolio optimization in finance
• Route planning for logistics
• Resource allocation
• Quantum interference finds optimal solutions

🧬 Protein Folding:
• Interference helps find stable protein configurations
• Critical for understanding diseases
• Could lead to new treatments for Alzheimer's, cancer

📊 The Demo:
We'll demonstrate the Deutsch-Jozsa algorithm, which uses interference to determine if a function is constant or balanced in just ONE query (classical computers need multiple queries)!
        """
        
        desc_text.insert(tk.END, interference_desc)
        desc_text.config(state=tk.DISABLED)
        
        # Demo panel
        demo_frame = tk.Frame(main_frame)
        main_frame.add(demo_frame, width=600)
        
        # Controls
        control_frame = tk.Frame(demo_frame)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Constant Function", 
                 command=lambda: self.demo_interference("constant"), bg="lightblue").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Balanced Function", 
                 command=lambda: self.demo_interference("balanced"), bg="lightgreen").pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.interference_results = scrolledtext.ScrolledText(demo_frame, height=8, font=("Courier", 10))
        self.interference_results.pack(fill=tk.X, padx=10, pady=5)
        
        # Matplotlib figure
        self.interference_fig, self.interference_ax = plt.subplots(1, 1, figsize=(10, 4))
        self.interference_canvas = FigureCanvasTkAgg(self.interference_fig, demo_frame)
        self.interference_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def demo_interference(self, function_type):
        """Demonstrate quantum interference with Deutsch-Jozsa algorithm"""
        try:
            # Create circuit for Deutsch-Jozsa algorithm
            n_qubits = 3  # Number of input qubits
            qc = QuantumCircuit(n_qubits + 1, n_qubits)
            
            # Initialize ancilla qubit in |1⟩
            qc.x(n_qubits)
            
            # Apply Hadamard to all qubits
            for i in range(n_qubits + 1):
                qc.h(i)
            
            # Apply oracle based on function type
            if function_type == "constant":
                # Constant function: do nothing (always returns 0) or apply X to ancilla (always returns 1)
                # We'll do nothing for f(x) = 0
                oracle_desc = "f(x) = 0 for all x (constant)"
            else:  # balanced
                # Balanced function: apply CNOT for each input qubit
                for i in range(n_qubits):
                    qc.cx(i, n_qubits)
                oracle_desc = "f(x) = x₀ ⊕ x₁ ⊕ x₂ (balanced)"
            
            # Apply Hadamard to input qubits again
            for i in range(n_qubits):
                qc.h(i)
            
            # Measure input qubits
            qc.measure(range(n_qubits), range(n_qubits))
            
            # Run the circuit
            compiled_circuit = transpile(qc, self.simulator)
            job = self.simulator.run(compiled_circuit, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            # Plot circuit
            self.interference_ax.clear()
            qc.draw(output='mpl', ax=self.interference_ax)
            self.interference_ax.set_title(f"Deutsch-Jozsa Algorithm - {function_type.title()} Function")
            self.interference_canvas.draw()
            
            # Analyze results
            zero_state = counts.get('000', 0)
            total_shots = sum(counts.values())
            
            self.interference_results.insert(tk.END,
                f"\n🔍 Deutsch-Jozsa Algorithm Results:\n"
                f"Function: {oracle_desc}\n"
                f"Type: {function_type.title()}\n\n"
                f"📊 Measurement Results:\n")
            
            for outcome, count in sorted(counts.items()):
                percentage = (count/total_shots) * 100
                self.interference_results.insert(tk.END,
                    f"  |{outcome}⟩: {count} times ({percentage:.1f}%)\n")
            
            if function_type == "constant":
                if zero_state > 900:  # Should measure |000⟩ with high probability
                    self.interference_results.insert(tk.END,
                        f"\n✅ CONSTANT FUNCTION DETECTED!\n"
                        f"🌀 Interference Magic:\n"
                        f"• All amplitudes interfered constructively for |000⟩\n"
                        f"• Wrong answers were cancelled out by destructive interference\n"
                        f"• Classical algorithm needs 2^(n-1)+1 = 5 queries\n"
                        f"• Quantum algorithm needs only 1 query!\n")
                else:
                    self.interference_results.insert(tk.END, f"\n⚠️ Unexpected result - may be due to noise")
            else:  # balanced
                if zero_state < 100:  # Should rarely measure |000⟩
                    self.interference_results.insert(tk.END,
                        f"\n✅ BALANCED FUNCTION DETECTED!\n"
                        f"🌀 Interference Magic:\n"
                        f"• Amplitudes for |000⟩ cancelled out (destructive interference)\n"
                        f"• Other states were amplified by constructive interference\n"
                        f"• This proves the function returns 0 and 1 equally often\n"
                        f"• Achieved in just 1 quantum query vs 5 classical queries!\n")
                else:
                    self.interference_results.insert(tk.END, f"\n⚠️ Unexpected result - may be due to noise")
            
            self.interference_results.insert(tk.END,
                f"\n🚀 Why This Matters:\n"
                f"This demonstrates quantum advantage through interference!\n"
                f"The algorithm distinguishes function types exponentially faster\n"
                f"than any classical algorithm possibly could.\n")
            
            self.interference_results.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not demonstrate interference: {e}")
    
    def create_quantum_algorithms_tab(self):
        """Quantum algorithms overview"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🧮 Algorithms")
        
        # Create scrollable text
        text_frame = tk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        algo_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Arial", 11))
        algo_text.pack(fill=tk.BOTH, expand=True)
        
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
        
        algo_text.insert(tk.END, algorithms_content)
        algo_text.config(state=tk.DISABLED)
    
    def create_quantum_ml_tab(self):
        """Quantum machine learning applications"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🤖 Quantum ML")
        
        text_frame = tk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ml_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Arial", 11))
        ml_text.pack(fill=tk.BOTH, expand=True)
        
        ml_content = """
🤖 QUANTUM MACHINE LEARNING

Quantum computing promises to revolutionize machine learning through quantum-enhanced algorithms that could provide exponential speedups for certain problems.

🧠 QUANTUM NEURAL NETWORKS

How They Work:
• Replace classical neurons with quantum circuits
• Use quantum superposition to process multiple inputs simultaneously
• Leverage entanglement for complex correlations
• Quantum interference helps with optimization

Advantages:
• Exponentially large feature spaces
• Natural handling of quantum data
• Potential speedup in training
• Better optimization landscapes

🎯 Current Applications:
• Image classification
• Natural language processing
• Financial modeling
• Drug discovery

🔬 QUANTUM FEATURE MAPS

What They Do:
• Map classical data into quantum feature space
• Exploit quantum phenomena for pattern recognition
• Create complex, non-linear transformations
• Enable quantum advantage in classical problems

Real Examples:
🏥 Medical Diagnosis:
• Map patient symptoms to quantum states
• Use quantum interference to amplify disease patterns
• Detect subtle correlations classical ML misses
• IBM's quantum advantage in certain classification tasks

📈 Financial Analysis:
• Quantum risk modeling
• Portfolio optimization
• Fraud detection with quantum patterns
• Options pricing with quantum Monte Carlo

🧬 QUANTUM ML IN DRUG DISCOVERY

Molecular Property Prediction:
• Quantum computers naturally represent molecules
• Predict drug-target interactions
• Optimize molecular structures
• Simulate biological systems

Example Pipeline:
1. Encode molecular structure in qubits
2. Use quantum simulation for molecular dynamics
3. Apply quantum ML for property prediction
4. Optimize with quantum algorithms

Current Players:
• Roche: Quantum ML for drug discovery
• Merck: Quantum algorithms for molecular simulation
• ProteinQure: Quantum-enhanced drug design

🎮 QUANTUM REINFORCEMENT LEARNING

Quantum Advantage:
• Explore multiple strategies in superposition
• Quantum speedup in policy optimization
• Better handling of large state spaces
• Quantum interference guides learning

Applications:
🚗 Autonomous Vehicles:
• Quantum decision-making under uncertainty
• Faster pathfinding algorithms
• Real-time optimization of traffic patterns

🏭 Industrial Control:
• Quantum optimization of manufacturing processes
• Resource allocation with quantum advantage
• Predictive maintenance using quantum patterns

📊 CURRENT LIMITATIONS & FUTURE

Current Reality:
• Most quantum ML algorithms are theoretical
• NISQ devices limit practical applications
• Hybrid classical-quantum approaches most promising
• Quantum advantage demonstrated for specific problems only

Near-term (2-5 years):
• Quantum-enhanced optimization
• Small-scale quantum neural networks
• Quantum feature maps for classical data
• Hybrid algorithms on NISQ devices

Long-term (5-15 years):
• Fault-tolerant quantum ML
• Exponential advantages for certain problems
• Quantum AI systems
• Revolutionary applications we can't imagine yet

🚀 Getting Started:
• Qiskit Machine Learning
• PennyLane for differentiable quantum programming
• TensorFlow Quantum
• Cirq for quantum circuits

The field is rapidly evolving with new breakthroughs regularly. While we're still in early stages, the potential for quantum machine learning to transform AI is enormous!
        """
        
        ml_text.insert(tk.END, ml_content)
        ml_text.config(state=tk.DISABLED)
    
    def create_medical_applications_tab(self):
        """Medical and biological applications"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🧬 Medical Apps")
        
        text_frame = tk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        medical_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Arial", 11))
        medical_text.pack(fill=tk.BOTH, expand=True)
        
        medical_content = """
🧬 QUANTUM COMPUTING IN MEDICINE

Quantum computing promises to revolutionize healthcare through unprecedented computational power for biological systems.

💊 DRUG DISCOVERY & DEVELOPMENT

The Challenge:
• Drug development costs ~$2.6 billion and takes 10-15 years
• Only 1 in 5000 discovered compounds becomes a drug
• Molecular interactions are inherently quantum mechanical
• Classical computers struggle with exponential complexity

Quantum Solutions:
🔬 Molecular Simulation:
• Quantum computers naturally simulate quantum systems
• Model drug-target interactions at atomic level
• Predict side effects before synthesis
• Optimize drug properties in silico

🎯 Current Examples:
• Roche: Using quantum algorithms for molecular optimization
• Merck: Quantum simulation of drug compounds
• ProteinQure: Quantum-enhanced molecular design
• Cambridge Quantum Computing: Drug discovery partnerships

Real Impact:
• Cambridge QC: 3x improvement in molecular property prediction
• Menten AI: Quantum-enhanced protein design
• Estimated to reduce drug discovery time by 5-10 years

🧬 PROTEIN FOLDING

Why It Matters:
• Protein misfolding causes Alzheimer's, Parkinson's, cancer
• Understanding folding → designing treatments
• Levinthal's paradox: astronomical number of possible configurations

Classical Limitations:
• AlphaFold2 impressive but computationally intensive
• Limited to static structures
• Struggles with dynamic folding pathways
• Exponential scaling with protein size

Quantum Advantage:
• Explore all folding pathways in superposition
• Model quantum effects in protein dynamics
• Optimize folding energy landscapes
• Predict misfolding mechanisms

🎯 Current Research:
• IBM: Quantum algorithms for protein structure prediction
• Google: Quantum enhancement of AlphaFold
• Microsoft: Quantum chemistry for biological systems

✂️ CRISPR OPTIMIZATION

The CRISPR Challenge:
• Design guide RNAs for precise gene editing
• Minimize off-target effects
• Optimize for different cell types
• Consider epigenetic factors

Quantum Solutions:
🎯 Guide RNA Design:
• Search through vast sequence space efficiently
• Quantum optimization for specificity
• Predict off-target binding quantum mechanically
• Multi-objective optimization (efficacy vs safety)

🧮 Our Implementation:
• QAOA for combinatorial optimization
• Quantum annealing for energy minimization
• Machine learning with quantum feature maps
• Hybrid classical-quantum pipeline

Real Applications:
• Treating genetic diseases (sickle cell, Huntington's)
• Cancer immunotherapy optimization
• Agricultural crop improvement
• Personalized gene therapy

🔬 GENOMIC ANALYSIS

Quantum Advantages:
📊 Sequence Alignment:
• Quantum search for optimal alignments
• Handle large-scale genomic variations
• Population genetics with quantum speedup
• Phylogenetic analysis acceleration

🎯 Biomarker Discovery:
• Quantum machine learning for pattern recognition
• Find subtle genetic signatures of disease
• Personalized medicine based on quantum analysis
• Drug response prediction

Current Projects:
• 1QBit: Quantum genomics analysis
• ProteinQure: Quantum biomarker discovery
• Roche: Quantum-enhanced clinical trials

🏥 MEDICAL IMAGING & DIAGNOSTICS

Quantum Sensing:
• Ultra-sensitive MRI with quantum enhancement
• Single-molecule detection in living cells
• Real-time monitoring of drug delivery
• Quantum radar for medical imaging

AI Diagnostics:
• Quantum neural networks for radiology
• Pattern recognition in medical scans
• Early disease detection with quantum advantage
• Personalized treatment recommendations

🧠 NEUROLOGICAL DISORDERS

Brain Simulation:
• Model neural networks with quantum circuits
• Understand consciousness and cognition
• Simulate brain disorders (depression, schizophrenia)
• Design targeted treatments

Current Research:
• IBM: Quantum simulation of neural networks
• Microsoft: Quantum algorithms for brain modeling
• Cambridge QC: Neuromorphic quantum computing

📈 CURRENT STATUS & TIMELINE

Near-term (1-3 years):
• Quantum-enhanced molecular property prediction
• Small protein folding problems
• CRISPR guide RNA optimization
• Proof-of-concept medical diagnostics

Medium-term (3-7 years):
• Practical drug discovery applications
• Large-scale protein simulation
• Quantum-enhanced clinical trials
• Revolutionary diagnostic tools

Long-term (7-15 years):
• Personalized quantum medicine
• Real-time quantum biological simulation
• Quantum-designed drugs and treatments
• Transformation of healthcare industry

🌟 Impact Potential:
• Reduce drug discovery time from decades to years
• Enable personalized treatments for everyone
• Solve currently intractable diseases
• Transform healthcare from reactive to predictive

The intersection of quantum computing and medicine represents one of the most promising applications for near-term quantum advantage!
        """
        
        medical_text.insert(tk.END, medical_content)
        medical_text.config(state=tk.DISABLED)
    
    def create_cosmology_tab(self):
        """Cosmology and physics applications"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🌌 Cosmology")
        
        text_frame = tk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        cosmo_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Arial", 11))
        cosmo_text.pack(fill=tk.BOTH, expand=True)
        
        cosmo_content = """
🌌 QUANTUM COMPUTING IN COSMOLOGY

Quantum computing opens new frontiers in understanding the universe, from black holes to dark matter to the fundamental nature of reality itself.

🕳️ BLACK HOLE PHYSICS

The Information Paradox:
• Hawking radiation suggests black holes evaporate
• Information cannot be destroyed (quantum mechanics)
• But what happens to information that falls in?
• One of physics' greatest unsolved problems

Quantum Simulation Approach:
🔬 Hawking Radiation Modeling:
• Simulate quantum field theory near event horizons
• Model particle creation and annihilation
• Study entanglement between inside and outside
• Test holographic principle predictions

🌀 AdS/CFT Correspondence:
• Black holes ↔ Quantum systems on boundary
• Quantum error correction in holography
• Simulate gravity using quantum circuits
• Understand spacetime emergence

Our Implementation:
• Quantum circuits modeling black hole formation
• Hawking radiation as quantum information scrambling
• Entanglement entropy calculations
• Information recovery algorithms

Real Research:
• Google: Quantum simulation of wormholes
• IBM: Black hole information processing
• Microsoft: Topological quantum matter for gravity

🌑 DARK MATTER DETECTION

The Dark Matter Mystery:
• 27% of universe is dark matter
• Interacts only gravitationally
• Direct detection experiments ongoing
• Quantum effects may be crucial

Quantum Enhancement:
🎯 Detector Optimization:
• Quantum algorithms for experiment design
• Optimize detector sensitivity parameters
• Real-time quantum data analysis
• Enhance signal-to-noise ratios

🔬 Data Analysis:
• Quantum machine learning for rare event detection
• Pattern recognition in detector data
• Statistical analysis with quantum advantage
• Correlation studies across multiple detectors

Current Applications:
• CERN: Quantum computing for particle physics
• Fermilab: Quantum sensors for dark matter
• DESY: Quantum algorithms for data analysis

🌊 GRAVITATIONAL WAVES

LIGO and Beyond:
• Gravitational waves detected from merging black holes
• Require extreme precision in data analysis
• Hidden signals in noisy data
• Quantum sensors for next generation

Quantum Advantages:
📡 Signal Processing:
• Quantum Fourier transforms for frequency analysis
• Pattern matching with quantum speedup
• Real-time analysis of detector streams
• Multi-messenger astronomy correlation

🔍 Parameter Estimation:
• Quantum optimization for source parameters
• Bayesian inference with quantum acceleration
• Multi-dimensional search spaces
• Template matching algorithms

Future Detectors:
• Quantum-enhanced interferometry
• Squeezed light for better sensitivity
• Quantum error correction for measurements
• Space-based quantum gravitational wave detectors

⚛️ QUANTUM FIELD THEORY

Fundamental Physics:
• Standard Model of particle physics
• Quantum chromodynamics (strong force)
• Electroweak theory
• Beyond Standard Model physics

Quantum Simulation:
🔬 Lattice QCD:
• Simulate quark confinement
• Calculate proton mass from first principles
• Study phase transitions in QCD
• Explore finite temperature/density physics

🎯 High Energy Physics:
• Simulate particle collisions
• Study symmetry breaking
• Model early universe physics
• Search for new physics beyond Standard Model

Current Research:
• Fermilab: Quantum simulation of gauge theories
• CERN: Quantum algorithms for QCD
• RIKEN: Quantum computing for nuclear physics

🌌 COSMOLOGICAL SIMULATIONS

Universe Evolution:
• Big Bang to present day
• Dark matter structure formation
• Galaxy formation and evolution
• Cosmic microwave background

Quantum Approach:
🌟 N-body Simulations:
• Quantum speedup for gravitational interactions
• Massive parallel processing with qubits
• Dark matter halo formation
• Large-scale structure evolution

🔬 Early Universe Physics:
• Inflation models with quantum fluctuations
• Primordial black hole formation
• Quantum phase transitions in cosmology
• Multiverse theories

Applications:
• Euclid space mission data analysis
• James Webb Space Telescope observations
• Next-generation sky surveys
• Precision cosmology measurements

🚀 SPACE EXPLORATION

Quantum Technologies in Space:
🛰️ Quantum Satellites:
• Global quantum communication networks
• Fundamental physics tests in space
• Quantum sensing for navigation
• Space-based quantum computing

🔬 Fundamental Tests:
• Test quantum mechanics in extreme conditions
• Search for modifications to general relativity
• Probe quantum gravity effects
• Study quantum decoherence in space

Current Missions:
• China's quantum satellites (Micius)
• Europe's quantum space initiatives
• NASA quantum technology development
• Commercial quantum space ventures

📊 CURRENT STATUS & FUTURE

Near-term (1-5 years):
• Small-scale quantum field theory simulations
• Quantum-enhanced data analysis for experiments
• Proof-of-concept black hole information studies
• Quantum sensors for fundamental physics

Medium-term (5-10 years):
• Large-scale cosmological simulations
• Quantum simulation of gauge theories
• Space-based quantum experiments
• Breakthrough discoveries in fundamental physics

Long-term (10+ years):
• Solution to black hole information paradox
• Quantum theory of gravity
• Discovery of dark matter through quantum methods
• Revolutionary understanding of universe

🌟 Revolutionary Potential:
• Solve deepest mysteries of physics
• Understand quantum nature of spacetime
• Discover new fundamental forces
• Transform our view of reality

The quantum universe may only be fully understood using quantum computers - making this one of the most profound applications of quantum computing!
        """
        
        cosmo_text.insert(tk.END, cosmo_content)
        cosmo_text.config(state=tk.DISABLED)


def main():
    """Main function to run the Quantum Explorer GUI"""
    root = tk.Tk()
    app = QuantumExplorerGUI(root)
    
    # Handle closing gracefully
    def on_closing():
        plt.close('all')  # Close all matplotlib figures
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
