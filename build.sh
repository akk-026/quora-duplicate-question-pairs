#!/usr/bin/env bash
# Force Python 3.10 (Render reads this from runtime.txt but this makes it explicit)
echo "Using Python version:"
python --version

# Install dependencies
pip install -r requirements.txt

