#!/usr/bin/env python3
"""
Quantum Hardware Explorer GUI

Minimal GUI that runs simple, hardware-ready quantum circuits on real devices
via Qiskit IBM Runtime. Only returns results from actual quantum computers.
"""

import sys

import matplotlib

# Use QtAgg backend for PyQt5 before importing other matplotlib modules
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
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

try:
    from qiskit import QuantumCircuit
    from qiskit.visualization import plot_histogram
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
except ImportError as e:
    print(f"Qiskit modules missing: {e}")
    print("Install with: pip install qiskit qiskit-ibm-runtime matplotlib PyQt5 pylatexenc")
    sys.exit(1)


class QuantumExplorerGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quantum Hardware Explorer")
        self.setGeometry(100, 100, 1200, 800)

        # IBM Runtime service (initialized on first use)
        self.service = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.create_introduction_tab()
        self.create_hardware_ready_tab()

        self.setStyleSheet(
            """
            QMainWindow { background-color: #f0f0f0; }
            QTabWidget::pane { border: 1px solid #cccccc; background-color: white; }
            QTabWidget::tab-bar { alignment: left; }
            QTabBar::tab { background-color: #e0e0e0; padding: 8px 16px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: white; border-bottom: 2px solid #007acc; }
            QPushButton { background-color: #007acc; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #005a9f; }
            QPushButton:pressed { background-color: #004080; }
            """
        )

    def create_introduction_tab(self) -> None:
        widget = QWidget()
        self.tab_widget.addTab(widget, "🚀 Introduction")
        layout = QVBoxLayout(widget)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Arial", 11))
        content = '''
<div style="font-family: Arial; font-size: 11pt;">
  <h2>🌟 Welcome to the Quantum Hardware Demo Explorer!</h2>
  <p>This app runs <b>real hardware</b> jobs on IBM Quantum using Qiskit Runtime.</p>
  <h3>🔑 Setup</h3>
  <p>Set these environment variables before running:</p>
  <pre>export QISKIT_IBM_TOKEN=your_token
export QISKIT_IBM_CHANNEL=ibm_quantum
export QISKIT_IBM_INSTANCE=ibm-q/open/main</pre>
  <p>Create an account at <a href="https://quantum.ibm.com/">quantum.ibm.com</a>.</p>
  <h3>🎯 What you can run</h3>
  <ul>
    <li>🌊 Single-qubit superposition</li>
    <li>🔗 Bell state (2-qubit entanglement)</li>
    <li>🌍 GHZ state (3-qubit entanglement)</li>
  </ul>
</div>
        '''
        text_edit.setHtml(content)
        layout.addWidget(text_edit)

    def create_hardware_ready_tab(self) -> None:
        widget = QWidget()
        self.tab_widget.addTab(widget, "🚀 Hardware-Ready Demos")
        main_layout = QHBoxLayout(widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left: description
        desc_widget = QWidget()
        desc_layout = QVBoxLayout(desc_widget)
        title = QLabel("🚀 Hardware-Ready Demos")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        desc_layout.addWidget(title)
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setFont(QFont("Arial", 10))
        desc.setPlainText(
            "Simple, shallow circuits to validate a real quantum device."
        )
        desc_layout.addWidget(desc)
        splitter.addWidget(desc_widget)

        # Right: controls and results
        demo_widget = QWidget()
        demo_layout = QVBoxLayout(demo_widget)

        buttons_layout = QVBoxLayout()

        btn_super = QPushButton("🌊 Single Qubit Superposition")
        btn_super.clicked.connect(self.run_superposition)
        buttons_layout.addWidget(btn_super)

        btn_bell = QPushButton("🔗 Bell State (2 Qubits)")
        btn_bell.clicked.connect(self.run_bell)
        buttons_layout.addWidget(btn_bell)

        btn_ghz = QPushButton("🌍 GHZ State (3 Qubits)")
        btn_ghz.clicked.connect(self.run_ghz)
        buttons_layout.addWidget(btn_ghz)

        demo_layout.addLayout(buttons_layout)

        self.results = QTextEdit()
        self.results.setFont(QFont("Courier", 10))
        self.results.setReadOnly(True)
        demo_layout.addWidget(self.results)

        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        demo_layout.addWidget(self.canvas)

        splitter.addWidget(demo_widget)
        splitter.setSizes([450, 750])

    # ---- IBM Runtime helpers ----
    def _ensure_service(self) -> None:
        if self.service is not None:
            return
        try:
            self.service = QiskitRuntimeService()
        except Exception as e:
            QMessageBox.critical(
                self,
                "IBM Runtime Not Configured",
                "IBM Runtime credentials not found.\n"
                "Set QISKIT_IBM_TOKEN, QISKIT_IBM_CHANNEL, QISKIT_IBM_INSTANCE and retry.\n\n"
                f"Original error: {e}",
            )
            raise

    def _select_backend(self, qubits: int):
        self._ensure_service()
        candidates = [
            b for b in self.service.backends()
            if getattr(b, "num_qubits", 0) >= qubits
        ]
        if not candidates:
            raise RuntimeError("No suitable backends available for required qubit count")
        # Choose first operational backend
        for b in candidates:
            try:
                if b.status().operational:
                    return b
            except Exception:
                continue
        return candidates[0]

    def _run_on_hardware(self, qc: QuantumCircuit, shots: int = 1024):
        backend = self._select_backend(qc.num_qubits)
        # Use a runtime Session bound to the selected backend; the service was
        # initialized in _ensure_service(), so default session discovery works.
        with Session(backend=backend) as session:
            # Create a Sampler bound to the active session context
            sampler = Sampler()
            job = sampler.run([qc], shots=shots)
            result = job.result()
            quasi = result.quasi_dists[0]
            bitlen = qc.num_clbits or qc.num_qubits
            counts = {}
            for k, v in quasi.items():
                try:
                    idx = int(k)
                except Exception:
                    try:
                        idx = int(k, 2)
                    except Exception:
                        continue
                key = format(idx, f"0{bitlen}b")
                counts[key] = int(round(v * shots))
            # Resolve backend name robustly across versions
            backend_name = getattr(backend, "name", None)
            if callable(backend_name):
                backend_name = backend_name()
            if not isinstance(backend_name, str):
                backend_name = getattr(backend, "backend_name", "backend")
            return counts, backend_name

    # ---- Demos ----
    def run_superposition(self) -> None:
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        self._execute_and_display("Single Qubit Superposition", qc)

    def run_bell(self) -> None:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        self._execute_and_display("Bell State", qc)

    def run_ghz(self) -> None:
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])
        self._execute_and_display("GHZ State", qc)

    def _execute_and_display(self, name: str, qc: QuantumCircuit) -> None:
        try:
            counts, backend_name = self._run_on_hardware(qc)
        except Exception as e:
            QMessageBox.critical(self, "Hardware Run Failed", str(e))
            return

        self.results.clear()
        self.results.append(f"🚀 {name}\n")
        self.results.append(f"Backend: {backend_name}\n")
        self.results.append("📊 Measurement Results:\n")
        total = max(1, sum(counts.values()))
        for outcome, count in sorted(counts.items()):
            pct = (count / total) * 100.0
            self.results.append(f"  |{outcome}⟩: {count} ({pct:.1f}%)")

        self.figure.clear()
        ax1 = self.figure.add_subplot(1, 2, 1)
        plot_histogram(counts, ax=ax1, title="Outcomes")
        ax2 = self.figure.add_subplot(1, 2, 2)
        qc.draw(output="mpl", ax=ax2)
        ax2.set_title("Circuit")
        self.canvas.draw()


def main() -> int:
    app = QApplication(sys.argv)
    explorer = QuantumExplorerGUI()
    explorer.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
