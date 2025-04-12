#!/bin/bash
# Apertis Installation Script
# This script helps you install and set up the Apertis LLM framework

set -e  # Exit on error

# Print colorful messages
print_green() {
    echo -e "\033[0;32m$1\033[0m"
}

print_blue() {
    echo -e "\033[0;34m$1\033[0m"
}

print_yellow() {
    echo -e "\033[0;33m$1\033[0m"
}

print_red() {
    echo -e "\033[0;31m$1\033[0m"
}

# Welcome message
print_blue "=================================================="
print_blue "  Welcome to Apertis LLM Installation"
print_blue "  A user-friendly multimodal LLM framework"
print_blue "=================================================="
echo ""

# Check Python version
print_yellow "Checking Python version..."
if command -v python3 &>/dev/null; then
    python_version=$(python3 --version)
    print_green "Found $python_version"
else
    print_red "Python 3 not found. Please install Python 3.8 or newer."
    exit 1
fi

# Check if virtual environment should be created
read -p "Do you want to create a virtual environment for Apertis? (recommended) [Y/n]: " create_venv
create_venv=${create_venv:-Y}

if [[ $create_venv =~ ^[Yy]$ ]]; then
    print_yellow "Checking for virtual environment tools..."
    
    if ! command -v pip3 &>/dev/null; then
        print_red "pip3 not found. Please install pip for Python 3."
        exit 1
    fi
    
    # Install virtualenv if not already installed
    if ! command -v virtualenv &>/dev/null; then
        print_yellow "Installing virtualenv..."
        pip3 install virtualenv
    fi
    
    # Create and activate virtual environment
    print_yellow "Creating virtual environment..."
    virtualenv venv
    
    print_yellow "Activating virtual environment..."
    source venv/bin/activate
    
    print_green "Virtual environment created and activated!"
fi

# Install dependencies
print_yellow "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

# Install the package in development mode
print_yellow "Installing Apertis in development mode..."
pip3 install -e .

# Create data directory if it doesn't exist
print_yellow "Setting up data directory..."
mkdir -p data

# Create a simple vocabulary file if it doesn't exist
if [ ! -f "data/vocab.json" ]; then
    print_yellow "Creating a sample vocabulary file..."
    cat > data/vocab.json << EOF
{
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "the": 4,
    "a": 5,
    "an": 6,
    "is": 7,
    "was": 8,
    "are": 9,
    "were": 10
}
EOF
fi

# Create a small model for testing
print_yellow "Creating a small test model..."
python3 -c "
import os
import torch
from apertis.model.core import create_apertis_model

# Create model directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Create a tiny model for testing
model = create_apertis_model(model_size='small', multimodal=True)
torch.save(model.state_dict(), 'models/test_model.pt')
"

# Installation complete
print_green "=================================================="
print_green "  Apertis installation complete!"
print_green "=================================================="
echo ""

# Print usage instructions
print_blue "Quick Start:"
echo ""
print_yellow "1. Start the web interface:"
echo "   python -m apertis.inference.interface --model-path models/test_model.pt --vocab-file data/vocab.json --web --multimodal"
echo ""
print_yellow "2. Start the command-line interface:"
echo "   python -m apertis.inference.interface --model-path models/test_model.pt --vocab-file data/vocab.json --multimodal"
echo ""
print_yellow "3. Run the example scripts:"
echo "   python examples/simple_chat.py"
echo ""

# If using virtual environment, remind the user to activate it
if [[ $create_venv =~ ^[Yy]$ ]]; then
    print_yellow "Remember to activate the virtual environment before using Apertis:"
    echo "   source venv/bin/activate"
    echo ""
fi

print_blue "For more information, see the documentation in docs/README.md"
print_blue "Enjoy using Apertis!"
