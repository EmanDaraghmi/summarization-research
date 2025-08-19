#!/bin/bash
# setup.sh - Environment setup script

echo "Setting up research environment..."
echo "=================================="

# Create virtual environment
python3 -m venv research-env
source research-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directory structure
mkdir -p data/raw data/processed models results logs

echo "Environment setup complete!"
echo "To activate: source research-env/bin/activate"