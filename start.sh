#!/bin/bash
set -e  # Exit on error

echo "Starting setup..."

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install optional dependencies for better performance if on CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing ninja and packaging for potential Flash Attention support..."
    pip install ninja packaging
fi

echo "Setup complete!"

# Run training
echo "Starting training..."
python3 train.py
