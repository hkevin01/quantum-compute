#!/usr/bin/env python3
"""
Environment Setup Script

This script sets up the quantum computing research environment
by installing dependencies and configuring the workspace.
"""

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return e.returncode


def install_quantum_packages():
    """Install required quantum computing packages."""
    packages = [
        "qiskit>=0.45.0",
        "qiskit-aer>=0.13.0", 
        "qiskit-ibm-runtime>=0.15.0",
        "qiskit-machine-learning>=0.7.0",
        "qiskit-optimization>=0.6.0",
        "qiskit-nature>=0.7.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "pandas>=1.3.0",
        "jupyter>=1.0.0",
        "pennylane>=0.32.0",
        "cirq>=1.0.0",
        "tensorflow-quantum>=0.7.0"
    ]
    
    print("📦 Installing quantum computing packages...")
    
    for package in packages:
        retcode = run_command(f"pip install {package}", f"Installing {package}")
        if retcode != 0:
            logger.warning(f"Failed to install {package}, continuing...")
    
    return True


def setup_jupyter_kernels():
    """Set up Jupyter kernels for quantum computing."""
    logger.info("Setting up Jupyter kernels...")
    
    commands = [
        ("python -m ipykernel install --user --name=quantum-research", 
         "Installing Jupyter kernel"),
        ("jupyter kernelspec list", "Listing available kernels")
    ]
    
    for command, description in commands:
        run_command(command, description)


def create_directories():
    """Create necessary project directories."""
    directories = [
        "data",
        "results", 
        "logs",
        "notebooks/examples",
        "tests/unit",
        "tests/integration",
        "docs/api",
        "docs/tutorials"
    ]
    
    logger.info("Creating project directories...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_config_files():
    """Create configuration files."""
    
    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Results and data
results/
logs/
*.log
data/raw/
*.pkl
*.h5

# IDE
.vscode/
.idea/
*.swp
*.swo

# Quantum backends
qiskit.conf
ibm-quantum/

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    # Create logging configuration
    logging_config = """
import logging
import os

def setup_logging(log_level=logging.INFO):
    \"\"\"Set up logging configuration.\"\"\"
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/quantum_research.log'),
            logging.StreamHandler()
        ]
    )
"""
    
    os.makedirs('src/config', exist_ok=True)
    with open('src/config/logging_config.py', 'w') as f:
        f.write(logging_config)
    
    logger.info("Created configuration files")


def verify_installation():
    """Verify that key packages are working."""
    logger.info("Verifying installation...")
    
    test_imports = [
        "import qiskit",
        "import qiskit_aer", 
        "import numpy",
        "import matplotlib.pyplot",
        "from qiskit import QuantumCircuit"
    ]
    
    for import_test in test_imports:
        try:
            exec(import_test)
            logger.info(f"✅ {import_test} - OK")
        except ImportError as e:
            logger.error(f"❌ {import_test} - FAILED: {e}")
            return False
    
    # Test basic quantum circuit creation
    try:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        logger.info("✅ Basic quantum circuit creation - OK")
    except Exception as e:
        logger.error(f"❌ Quantum circuit test failed: {e}")
        return False
    
    return True


def main():
    """Main setup function."""
    print("🚀 Quantum Computing Research Environment Setup")
    print("=" * 60)
    
    logger.info("Starting environment setup...")
    
    # Create directories
    create_directories()
    
    # Install packages
    if not install_quantum_packages():
        logger.error("Package installation failed")
        return 1
    
    # Setup Jupyter
    setup_jupyter_kernels()
    
    # Create config files
    create_config_files()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'jupyter notebook' to start Jupyter")
        print("2. Execute example scripts in the scripts/ directory")
        print("3. Check out the notebooks/ directory for tutorials")
        return 0
    else:
        print("\n❌ Environment setup failed verification")
        print("Please check the logs and try installing missing packages manually")
        return 1


if __name__ == "__main__":
    exit(main())
