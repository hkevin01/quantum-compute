#!/usr/bin/env python3
"""
Quantum Hardware Explorer GUI (hardware-only)

Minimal PyQt5 app that runs shallow circuits on IBM Quantum hardware
via Qiskit IBM Runtime (Sampler). Only real-device results are shown.
"""

import sys
from typing import Optional

import matplotlib

# Select QtAgg for PyQt5 before importing Matplotlib UI modules
matplotlib.use("QtAgg")

from matplotlib.backends import backend_qtagg as _bqtagg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

FigureCanvas = _bqtagg.FigureCanvasQTAgg

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
    print(
        "Install with: pip install qiskit qiskit-ibm-runtime "
        "matplotlib PyQt5 pylatexenc"
    )
    sys.exit(1)


class QuantumExplorerGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quantum Hardware Explorer")
        self.setGeometry(100, 100, 1200, 800)

        # Lazy IBM Runtime service
        self.service: Optional[QiskitRuntimeService] = None

        # Predeclare widgets for type checkers
        self.results: Optional[QTextEdit] = None
        self.fig: Optional[Figure] = None
        self.canvas: Optional[FigureCanvas] = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._apply_styles()
        self._init_tab_intro()
        self._init_tab_hardware()

    def _apply_styles(self) -> None:
        css = (
            "QMainWindow { background-color: #f7f7f7; }\n"
            "QTabWidget::pane { border: 1px solid #ccc; background-color: "
            "#fff; }\n"
            "QTabWidget::tab-bar { alignment: left; }\n"
            "QTabBar::tab { background-color: #e8e8e8; padding: 6px 12px; "
            "margin-right: 2px; border-top-left-radius: 4px; "
            "border-top-right-radius: 4px; }\n"
            "QTabBar::tab:selected { background-color: #fff; border-bottom: "
            "2px solid #007acc; }\n"
            "QPushButton { background-color: #007acc; color: #fff; border: "
            "none; padding: 8px 14px; border-radius: 4px; font-weight: "
            "600; }\n"
            "QPushButton:hover { background-color: #0062b8; }\n"
            "QPushButton:pressed { background-color: #004f94; }\n"
        )
        self.setStyleSheet(css)

    def _init_tab_intro(self) -> None:
        w = QWidget()
        self.tabs.addTab(w, "Introduction")
        v = QVBoxLayout(w)

        t = QTextEdit()
        t.setReadOnly(True)
        t.setFont(QFont("Arial", 11))
        html = (
            "<div style=\"font-family: Arial; font-size: 11pt;\">\n"
            "<h2>Welcome to the Quantum Hardware Demo Explorer</h2>\n"
            "<p>This app submits jobs to <b>real IBM Quantum devices</b> "
            "using Qiskit Runtime.</p>\n"
            "<h3>Setup</h3>\n"
            "<p>Export these variables first:</p>\n"
            "<pre>export QISKIT_IBM_TOKEN=your_token\n"
            "export QISKIT_IBM_CHANNEL=ibm_quantum\n"
            "export QISKIT_IBM_INSTANCE=ibm-q/open/main</pre>\n"
            "<p>Create an account at <a href=\"https://quantum.ibm.com/\">"
            "quantum.ibm.com</a>.</p>\n"
            "</div>\n"
        )
        t.setHtml(html)
        v.addWidget(t)

    def _init_tab_hardware(self) -> None:
        w = QWidget()
        self.tabs.addTab(w, "Hardware-Ready Demos")
        h = QHBoxLayout(w)

        # Left controls
        left = QWidget()
        lv = QVBoxLayout(left)
        title = QLabel("Hardware-Ready Demos")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        lv.addWidget(title)

        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setFont(QFont("Arial", 10))
        desc.setPlainText(
            "Run shallow circuits: superposition, Bell, GHZ on hardware."
        )
        lv.addWidget(desc)

        b1 = QPushButton("Single Qubit Superposition")
        b1.clicked.connect(self.run_superposition)
        lv.addWidget(b1)

        b2 = QPushButton("Bell State (2 Qubits)")
        b2.clicked.connect(self.run_bell)
        lv.addWidget(b2)

        b3 = QPushButton("GHZ State (3 Qubits)")
        b3.clicked.connect(self.run_ghz)
        lv.addWidget(b3)

        # Right results
        right = QWidget()
        rv = QVBoxLayout(right)

        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setFont(QFont("Courier New", 10))
        rv.addWidget(self.results)

        self.fig = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.fig)
        rv.addWidget(self.canvas)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([400, 800])
        h.addWidget(splitter)

    # ---------- IBM Runtime helpers ----------
    def _ensure_service(self) -> None:
        if self.service is not None:
            return
        try:
            self.service = QiskitRuntimeService()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "IBM Runtime Not Configured",
                "IBM Runtime credentials not found.\n"
                "Set QISKIT_IBM_TOKEN, QISKIT_IBM_CHANNEL, "
                "QISKIT_IBM_INSTANCE and retry.\n\n"
                f"Original error: {e}",
            )
            raise

    def _pick_backend(self, qubits: int):
        self._ensure_service()
        try:
            assert self.service is not None
            backends = list(self.service.backends())
        except Exception:  # noqa: BLE001
            backends = []
        cands = [b for b in backends if getattr(b, "num_qubits", 0) >= qubits]
        if not cands:
            raise RuntimeError("No suitable backends for required qubit count")
        for b in cands:
            try:
                if b.status().operational:
                    return b
            except Exception:  # noqa: BLE001
                continue
        return cands[0]

    def _run(self, qc: QuantumCircuit, shots: int = 1024):
        backend = self._pick_backend(qc.num_qubits)
        with Session(backend=backend):
            sampler = Sampler(options={"shots": shots})
            job = sampler.run([qc])
            result = job.result()
        quasi = result.quasi_dists[0]
        bitlen = qc.num_clbits or qc.num_qubits
        counts = {}
        for k, v in quasi.items():
            try:
                idx = int(k)
            except Exception:  # noqa: BLE001
                try:
                    idx = int(k, 2)
                except Exception:  # noqa: BLE001
                    continue
            key = format(idx, f"0{bitlen}b")
            counts[key] = int(round(float(v) * shots))

        name = getattr(backend, "name", None)
        backend_name = name() if callable(name) else name
        if not isinstance(backend_name, str):
            backend_name = getattr(backend, "backend_name", "backend")
        return counts, backend_name

    # ---------- Demo actions ----------
    def run_superposition(self) -> None:
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        self._execute("Single Qubit Superposition", qc)

    def run_bell(self) -> None:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        self._execute("Bell State", qc)

    def run_ghz(self) -> None:
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])
        self._execute("GHZ State", qc)

    def _execute(self, title: str, qc: QuantumCircuit) -> None:
        try:
            counts, backend_name = self._run(qc)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Hardware Run Failed", str(e))
            return

        assert self.results is not None
        assert self.fig is not None
        assert self.canvas is not None

        self.results.clear()
        self.results.append(f"Job: {title}\n")
        self.results.append(f"Backend: {backend_name}\n")
        self.results.append("Results:\n")
        total = max(1, sum(counts.values()))
        for bit, cnt in sorted(counts.items()):
            pct = 100.0 * cnt / total
            self.results.append(f"  |{bit}⟩: {cnt} ({pct:.1f}%)")

        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        plot_histogram(counts, ax=ax1, title="Outcomes")
        ax2 = self.fig.add_subplot(1, 2, 2)
        qc.draw(output="mpl", ax=ax2)
        ax2.set_title("Circuit")
        self.canvas.draw()


def main() -> int:
    app = QApplication(sys.argv)
    win = QuantumExplorerGUI()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
