# Quantum Computing Explorer

This repository contains quantum computing demonstrations and algorithms designed to run on real quantum hardware, focusing on NISQ-era applications.

## 🚀 Hardware-Ready Quantum Demos

This project provides a collection of quantum circuits that are optimized for today's quantum computers. These examples are:
- **Simple**: Using 1-4 qubits.
- **Shallow**: Low gate depth to minimize noise.
- **Fundamental**: Demonstrating core quantum phenomena like superposition and entanglement.
- **Practical**: Including NISQ algorithms like VQE and QAOA.

## 🛠 Getting Started

### Easy Setup (Recommended)
Use the automated setup script:
```bash
# Make script executable and run
chmod +x run.sh
./run.sh
```

This will automatically:
- Check Python version (3.7+ required)
- Create a virtual environment
- Install all necessary dependencies
- Launch the Quantum Explorer GUI

### Manual Setup
If you prefer manual installation:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib numpy PyQt5 pylatexenc
```

### Quick Start - Interactive GUI
Launch the interactive Quantum Computing Explorer:
```bash
# Using the run script (recommended)
./run.sh gui

# Or manually
python launch_gui.py
```

### Other Run Options
```bash
./run.sh demos        # Interactive command-line demonstrations
./run.sh test          # Run quantum computing tests  
./run.sh examples      # Basic quantum examples
./run.sh nisq          # NISQ-optimized quantum algorithms
./run.sh hardware      # Hardware-ready quantum demos
./run.sh help          # Show all options
```

The GUI provides:
- 🌊 **Superposition demonstrations** with live circuit visualization
- 🔗 **Entanglement experiments** showing quantum correlations
- 🔬 **NISQ algorithms** optimized for real quantum computers
- 🚀 **Hardware-ready demos** perfect for testing quantum devices

### Project Structure
- `gui/` - Interactive GUI application
- `examples/` - Runnable quantum computing examples
- `notebooks/` - Jupyter notebooks for learning
- `docs/` - Documentation and guides

## 📚 Learning Resources

- `docs/quantum_computing_guide.md`: A guide to fundamental quantum computing concepts.
- `docs/qiskit_guide.md`: A guide to using the Qiskit framework.
- `docs/space_medical_quantum_applications.md`: Documentation of advanced quantum applications (planned for future development).
- `notebooks/quantum_basics_tutorial.ipynb`: An interactive Jupyter notebook for learning quantum basics.

---

*Note: These examples are designed for educational and experimental purposes on real quantum hardware. For advanced applications like medical genomics and cosmology, see the documentation in `docs/space_medical_quantum_applications.md`.*
