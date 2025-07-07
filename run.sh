#!/bin/bash

# Quantum Computing Research Projects - Run Script
# This script sets up the environment and launches the quantum computing explorer

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project info
PROJECT_NAME="Quantum Computing Explorer"
PROJECT_VERSION="1.0.0"

# Function to print colored output
print_header() {
    echo -e "${PURPLE}================================================${NC}"
    echo -e "${PURPLE}🚀 $PROJECT_NAME v$PROJECT_VERSION${NC}"
    echo -e "${PURPLE}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${CYAN}🔄 $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    print_step "Checking Python installation..."
    
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python 3.7+ first."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
        print_error "Python 3.7+ is required. Found version $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION found"
}

# Function to check and create virtual environment
setup_venv() {
    print_step "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    print_success "Virtual environment activated"
}

# Function to install dependencies
install_dependencies() {
    print_step "Installing dependencies..."
    
    # Try full requirements first, fall back to core if needed
    if [ -f "requirements.txt" ]; then
        print_info "Installing from requirements.txt..."
        if pip install -r requirements.txt; then
            print_success "All dependencies installed successfully"
        else
            print_warning "Some dependencies failed. Trying core requirements..."
            if [ -f "requirements-core.txt" ]; then
                pip install -r requirements-core.txt
                print_success "Core dependencies installed successfully"
                print_info "Some optional packages may not be available"
            else
                print_warning "Falling back to manual installation..."
                pip install PyQt5 qiskit qiskit-aer matplotlib numpy pylatexenc
                print_success "Essential dependencies installed"
            fi
        fi
    else
        print_warning "requirements.txt not found. Installing core dependencies..."
        pip install PyQt5 qiskit qiskit-aer matplotlib numpy scipy pylatexenc
        print_success "Core dependencies installed"
    fi
}

# Function to run tests
run_tests() {
    print_step "Running quantum computing tests..."
    
    if [ -f "scripts/test_quantum.py" ]; then
        $PYTHON_CMD scripts/test_quantum.py
        print_success "Tests completed"
    else
        print_warning "Test script not found, skipping tests"
    fi
}

# Function to launch GUI
launch_gui() {
    print_step "Launching Quantum Computing Explorer GUI..."
    
    if [ -f "launch_gui.py" ]; then
        $PYTHON_CMD launch_gui.py
    else
        print_error "launch_gui.py not found!"
        exit 1
    fi
}

# Function to run interactive demos
run_demos() {
    print_step "Running interactive quantum demonstrations..."
    
    if [ -f "examples/interactive_demos.py" ]; then
        $PYTHON_CMD examples/interactive_demos.py
    else
        print_error "Interactive demos not found!"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo -e "${BLUE}Usage: ./run.sh [OPTION]${NC}"
    echo ""
    echo "Options:"
    echo "  gui          Launch the PyQt5 GUI (default)"
    echo "  demos        Run interactive command-line demonstrations"
    echo "  test         Run quantum computing tests"
    echo "  setup        Set up environment and install dependencies only"
    echo "  clean        Remove virtual environment"
    echo "  examples     Run basic quantum examples"
    echo "  crispr       Run CRISPR optimization demo"
    echo "  blackhole    Run black hole simulation"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh              # Launch GUI (default)"
    echo "  ./run.sh gui          # Launch GUI explicitly"
    echo "  ./run.sh demos        # Run interactive demos"
    echo "  ./run.sh test         # Run tests"
    echo "  ./run.sh setup        # Setup only"
}

# Function to run specific examples
run_examples() {
    print_step "Running basic quantum examples..."
    
    if [ -f "examples/basic_quantum_examples.py" ]; then
        $PYTHON_CMD examples/basic_quantum_examples.py
    else
        print_error "Basic examples not found!"
        exit 1
    fi
}

# Function to run CRISPR demo
run_crispr() {
    print_step "Running CRISPR optimization demonstration..."
    
    if [ -f "scripts/run_crispr_optimizer.py" ]; then
        $PYTHON_CMD scripts/run_crispr_optimizer.py
    else
        print_error "CRISPR script not found!"
        exit 1
    fi
}

# Function to run black hole simulation
run_blackhole() {
    print_step "Running black hole simulation..."
    
    if [ -f "scripts/run_black_hole_sim.py" ]; then
        $PYTHON_CMD scripts/run_black_hole_sim.py
    else
        print_error "Black hole simulation script not found!"
        exit 1
    fi
}

# Function to clean environment
clean_env() {
    print_step "Cleaning virtual environment..."
    
    if [ -d "venv" ]; then
        rm -rf venv
        print_success "Virtual environment removed"
    else
        print_info "No virtual environment to remove"
    fi
}

# Main execution
main() {
    print_header
    
    # Parse command line arguments
    case "${1:-gui}" in
        "gui"|"")
            check_python
            setup_venv
            install_dependencies
            launch_gui
            ;;
        "demos")
            check_python
            setup_venv
            install_dependencies
            run_demos
            ;;
        "test")
            check_python
            setup_venv
            install_dependencies
            run_tests
            ;;
        "setup")
            check_python
            setup_venv
            install_dependencies
            print_success "Environment setup complete!"
            ;;
        "examples")
            check_python
            setup_venv
            install_dependencies
            run_examples
            ;;
        "crispr")
            check_python
            setup_venv
            install_dependencies
            run_crispr
            ;;
        "blackhole")
            check_python
            setup_venv
            install_dependencies
            run_blackhole
            ;;
        "clean")
            clean_env
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Trap to handle script interruption
trap 'echo -e "\n${YELLOW}Script interrupted by user${NC}"; exit 130' INT

# Check if script is being run from correct directory
if [ ! -f "README.md" ] || [ ! -f "launch_gui.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Run main function
main "$@"
