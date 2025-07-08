#!/usr/bin/env python3
"""
Quantum Hardware Explorer GUI.

An interactive GUI for exploring quantum algorithms designed for real hardware.
"""

import os
import sys

import matplotlib

# Set the backend before importing other matplotlib modules
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PyQt5 not found. Install with: pip install PyQt5")
    sys.exit(1)

# Add the examples directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

try:
    # Import the hardware-ready and NISQ example modules
    import hardware_ready_demo
    import nisq_examples
    from qiskit.visualization import plot_histogram
    from qiskit_aer import AerSimulator
except ImportError as e:
    print(f"Warning: Could not import quantum modules: {e}")
    print("Make sure Qiskit is installed: pip install qiskit qiskit-aer")
    sys.exit(1)


class QuantumExplorerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Hardware Explorer")
        self.setGeometry(100, 100, 1200, 800)

        self.simulator = AerSimulator()
        self.nisq_examples = nisq_examples.NISQQuantumExamples()
        self.hardware_demos = hardware_ready_demo.HardwareReadyQuantumDemos()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.create_introduction_tab()
        self.create_nisq_algorithms_tab()
        self.create_hardware_ready_tab()

        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QTabWidget::pane { border: 1px solid #cccccc; background-color: white; }
            QTabWidget::tab-bar { alignment: left; }
            QTabBar::tab { background-color: #e0e0e0; padding: 8px 16px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: white; border-bottom: 2px solid #007acc; }
            QPushButton { background-color: #007acc; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #005a9f; }
            QPushButton:pressed { background-color: #004080; }
        """)

    def create_introduction_tab(self):
        widget = QWidget()
        self.tab_widget.addTab(widget, "🚀 Introduction")
        layout = QVBoxLayout(widget)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Arial", 11))
        content = '''
<div style="font-family: Arial; font-size: 11pt;">
    <h2>🌟 Welcome to the Quantum Hardware Demo Explorer!</h2>
    <p>This interactive application demonstrates quantum algorithms that are
    designed to run on <b>real quantum computers</b>.</p>

    <h3>🔬 What is NISQ?</h3>
    <p>We are currently in the <b>NISQ (Noisy Intermediate-Scale Quantum)</b>
    era. This means that today's quantum computers have a limited number of
    qubits (typically 50-1000) and are susceptible to noise, which can
    corrupt calculations.</p>

    <p>The algorithms in this explorer are specifically designed to be:</p>
    <ul>
        <li><b>Shallow</b>: They have a low number of sequential operations
        (low "depth") to reduce the impact of noise.</li>
        <li><b>Efficient</b>: They use a small number of qubits.</li>
        <li><b>Hybrid</b>: They often combine classical and quantum computation to
        solve practical problems.</li>
    </ul>

    <h3>🎯 Explore the Tabs:</h3>
    <ul>
        <li><b>NISQ Algorithms</b>: Discover algorithms like VQE and QAOA that
        are at the forefront of quantum research.</li>
        <li><b>Hardware-Ready Demos</b>: Run the most basic quantum circuits,
        like Bell states, to verify the fundamental principles of quantum
        mechanics on a real device.</li>
    </ul>
    <p>Ready to explore the quantum realm? Click through the tabs to see these
    algorithms in action! 🚀</p>
</div>
        '''
        text_edit.setHtml(content)
        layout.addWidget(text_edit)

    def create_nisq_algorithms_tab(self):
        widget = QWidget()
        self.tab_widget.addTab(widget, "🔬 NISQ Algorithms")
        main_layout = QHBoxLayout(widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        desc_widget = QWidget()
        desc_layout = QVBoxLayout(desc_widget)
        desc_label = QLabel("🔬 NISQ-Era Quantum Algorithms")
        desc_label.setFont(QFont("Arial", 16, QFont.Bold))
        desc_layout.addWidget(desc_label)
        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setFont(QFont("Arial", 10))
        nisq_desc = """
🎯 WHAT IS NISQ?

NISQ = Noisy Intermediate-Scale Quantum
• Current quantum computers (50-1000 qubits)
• High error rates, limited gate depth
• No error correction yet
• Hybrid classical-quantum algorithms

🚀 NISQ-OPTIMIZED ALGORITHMS:

1. 🎲 QUANTUM RANDOM NUMBER GENERATOR
   • True quantum randomness from superposition
   • Perfect for cryptography and simulations

2. ⚛️ VARIATIONAL QUANTUM EIGENSOLVER (VQE)
   • Find ground state energies of molecules
   • Hybrid optimization approach

3. 🔀 QUANTUM APPROXIMATE OPTIMIZATION (QAOA)
   • Solve combinatorial optimization problems
   • Max-Cut, traveling salesman, etc.

🔧 WHY THESE WORK ON NISQ DEVICES:
• Shallow circuits (low gate depth)
• Minimal qubits (2-10 typically)
• Noise-tolerant algorithms

💡 Try the interactive demos to see these algorithms in action!
        """
        desc_text.setPlainText(nisq_desc)
        desc_layout.addWidget(desc_text)
        splitter.addWidget(desc_widget)

        demo_widget = QWidget()
        demo_layout = QVBoxLayout(demo_widget)
        control_layout = QVBoxLayout()

        rng_btn = QPushButton("🎲 Quantum Random Numbers")
        rng_btn.clicked.connect(self.demo_quantum_rng)
        control_layout.addWidget(rng_btn)

        vqe_btn = QPushButton("⚛️ VQE for H2 Molecule")
        vqe_btn.clicked.connect(self.demo_vqe)
        control_layout.addWidget(vqe_btn)

        qaoa_btn = QPushButton("🔀 QAOA Optimization")
        qaoa_btn.clicked.connect(self.demo_qaoa)
        control_layout.addWidget(qaoa_btn)

        demo_layout.addLayout(control_layout)

        self.nisq_results = QTextEdit()
        self.nisq_results.setFont(QFont("Courier", 9))
        self.nisq_results.setReadOnly(True)
        demo_layout.addWidget(self.nisq_results)

        self.nisq_figure = Figure(figsize=(10, 6))
        self.nisq_canvas = FigureCanvas(self.nisq_figure)
        demo_layout.addWidget(self.nisq_canvas)

        splitter.addWidget(demo_widget)
        splitter.setSizes([400, 800])

    def create_hardware_ready_tab(self):
        widget = QWidget()
        self.tab_widget.addTab(widget, "🚀 Hardware-Ready Demos")
        main_layout = QHBoxLayout(widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        desc_widget = QWidget()
        desc_layout = QVBoxLayout(desc_widget)
        desc_label = QLabel("🚀 Hardware-Ready Demos")
        desc_label.setFont(QFont("Arial", 16, QFont.Bold))
        desc_layout.addWidget(desc_label)
        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setFont(QFont("Arial", 10))
        hardware_desc = """
🎯 WHAT ARE HARDWARE-READY DEMOS?

These are the simplest, most fundamental quantum circuits. They are perfect for testing a real quantum computer's capabilities.

✅ Key Features:
• Minimal Qubits (1-3)
• Minimal Depth (1-2 gates)
• Demonstrates core quantum phenomena

🚀 THE DEMOS:

1. 🌊 SINGLE QUBIT SUPERPOSITION
   • The "Hello, World!" of quantum computing
   • Puts a single qubit into a 50/50 state of 0 and 1

2. 🔗 BELL STATE (ENTANGLEMENT)
   • Creates a mysterious link between two qubits
   • Measuring one instantly affects the other

3. 🌍 GHZ STATE (3-QUBIT ENTANGLEMENT)
   • Entangles three qubits into a single state
   • Used in quantum error correction and networking

💡 Try the interactive demos to create and measure these fundamental quantum states!
        """
        desc_text.setPlainText(hardware_desc)
        desc_layout.addWidget(desc_text)
        splitter.addWidget(desc_widget)

        demo_widget = QWidget()
        demo_layout = QVBoxLayout(demo_widget)
        control_layout = QVBoxLayout()

        superposition_btn = QPushButton("🌊 Single Qubit Superposition")
        superposition_btn.clicked.connect(self.demo_single_qubit_superposition)
        control_layout.addWidget(superposition_btn)

        bell_state_btn = QPushButton("🔗 Bell State (Entanglement)")
        bell_state_btn.clicked.connect(self.demo_bell_state)
        control_layout.addWidget(bell_state_btn)

        ghz_state_btn = QPushButton("🌍 GHZ State (3-Qubit Entanglement)")
        ghz_state_btn.clicked.connect(self.demo_ghz_state)
        control_layout.addWidget(ghz_state_btn)

        demo_layout.addLayout(control_layout)

        self.hardware_results = QTextEdit()
        self.hardware_results.setFont(QFont("Courier", 10))
        self.hardware_results.setReadOnly(True)
        demo_layout.addWidget(self.hardware_results)

        self.hardware_figure = Figure(figsize=(12, 6))
        self.hardware_canvas = FigureCanvas(self.hardware_figure)
        demo_layout.addWidget(self.hardware_canvas)

        splitter.addWidget(demo_widget)
        splitter.setSizes([450, 750])

    def _run_demo(self, circuit_func, name, results_widget, figure, canvas):
        try:
            qc, counts, description = circuit_func()

            results_widget.clear()
            results_widget.append(f"🚀 RUNNING: {name}\n")
            results_widget.append(f"{description}\n")
            results_widget.append("📊 Measurement Results (1024 shots):")

            sorted_counts = sorted(counts.items())
            for outcome, count in sorted_counts:
                percentage = (count / 1024) * 100
                clean_outcome = outcome.replace(" ", "")
                results_widget.append(f"  |{clean_outcome}⟩: {count} times ({percentage:.1f}%)")

            figure.clear()
            ax1 = figure.add_subplot(1, 2, 1)
            plot_histogram(counts, ax=ax1, title="Measurement Outcomes")

            ax2 = figure.add_subplot(1, 2, 2)
            qc.draw(output='mpl', ax=ax2)
            ax2.set_title("Quantum Circuit")

            canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not run demo '{name}': {e}")

    def demo_quantum_rng(self):
        _, qc, counts, desc = self.nisq_examples.quantum_random_number_generator()
        self._display_nisq_results("Quantum Random Number Generator", qc, counts, desc)

    def demo_vqe(self):
        qc, result, desc = self.nisq_examples.vqe_h2_molecule()
        self._display_nisq_results("VQE for H2", qc, result, desc)

    def demo_qaoa(self):
        qc, result, desc = self.nisq_examples.qaoa_max_cut()
        self._display_nisq_results("QAOA Max-Cut", qc, result, desc)

    def _display_nisq_results(self, name, qc, result, description):
        self.nisq_results.clear()
        self.nisq_results.append(f"🚀 {name}\n")
        self.nisq_results.append(f"{description}\n")
        self.nisq_results.append("📊 Results:")
        self.nisq_results.append(str(result))

        self.nisq_figure.clear()
        ax = self.nisq_figure.add_subplot(111)
        qc.draw(output='mpl', ax=ax)
        self.nisq_canvas.draw()

    def demo_single_qubit_superposition(self):
        qc, counts, desc = self.hardware_demos.single_qubit_superposition()
        self._display_hardware_results("Single Qubit Superposition", qc, counts, desc)

    def demo_bell_state(self):
        qc, counts, desc = self.hardware_demos.create_bell_state()
        self._display_hardware_results("Bell State", qc, counts, desc)

    def demo_ghz_state(self):
        qc, counts, desc = self.hardware_demos.create_ghz_state()
        self._display_hardware_results("GHZ State", qc, counts, desc)

    def _display_hardware_results(self, name, qc, counts, description):
        self.hardware_results.clear()
        self.hardware_results.append(f"🚀 {name}\n")
        self.hardware_results.append(f"{description}\n")
        self.hardware_results.append("📊 Measurement Results:")

        sorted_counts = sorted(counts.items())
        for outcome, count in sorted_counts:
            percentage = (sum(counts.values()) / 100)
            self.hardware_results.append(f"  |{outcome}⟩: {count} times ({count/percentage:.1f}%)")

        self.hardware_figure.clear()
        ax1 = self.hardware_figure.add_subplot(1, 2, 1)
        plot_histogram(counts, ax=ax1, title="Outcomes")
        ax2 = self.hardware_figure.add_subplot(1, 2, 2)
        qc.draw(output='mpl', ax=ax2)
        ax2.set_title("Circuit")
        self.hardware_canvas.draw()


def main():
    """Main function to launch the Quantum Explorer GUI"""
    app = QApplication(sys.argv)
    explorer = QuantumExplorerGUI()
    explorer.show()
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())
