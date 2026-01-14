#!/bin/bash
# Test script for luminal_python
# Builds the extension and runs pytest

set -e  # Exit on error

echo "Building extension with maturin..."
maturin develop

echo ""
echo "Running tests..."
pytest -q "$@"
