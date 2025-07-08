"""
Setup configuration for Quantum Computing Research Projects
"""

import os

from setuptools import find_packages, setup


# Read the README file for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="quantum-research",
    version="0.1.0",
    author="Quantum Research Team",
    author_email="research@quantum-computing.org",
    description="Quantum computing demonstrations and algorithms for NISQ-era hardware",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/quantum-research/quantum-compute",
    
    packages=find_packages(),
    include_package_data=True,
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "molecular": [
            "rdkit-pypi>=2023.3.0",
            "openmm>=8.0.0",
            "pyscf>=2.3.0",
            "chempy>=0.8.0",
        ],
        "ml": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "scikit-learn>=1.3.0",
            "xgboost>=1.7.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "quantum-setup=scripts.setup_environment:main",
        ],
    },
    
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.json"],
        "docs": ["*.rst", "*.md"],
        "notebooks": ["*.ipynb"],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/quantum-research/quantum-compute/issues",
        "Source": "https://github.com/quantum-research/quantum-compute",
        "Documentation": "https://quantum-research.readthedocs.io/",
    },
    
    keywords=[
        "quantum computing",
        "quantum algorithms", 
        "NISQ algorithms",
        "quantum machine learning",
        "qiskit",
        "variational quantum algorithms",
        "QAOA",
        "VQE",
        "quantum optimization",
        "quantum simulation",
        "quantum hardware",
    ],
    
    zip_safe=False,
)
