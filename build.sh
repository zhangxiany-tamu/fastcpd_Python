#!/bin/bash
# Build script for fastcpd-python

set -e  # Exit on error

echo "========================================="
echo "Building fastcpd-python"
echo "========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check for required tools
command -v cmake >/dev/null 2>&1 || { echo "Error: cmake is required but not installed."; exit 1; }
command -v gfortran >/dev/null 2>&1 || { echo "Warning: gfortran not found. GARCH models may not work."; }

echo "CMake version: $(cmake --version | head -1)"

# Check for Armadillo
if pkg-config --exists armadillo; then
    echo "Armadillo found: $(pkg-config --modversion armadillo)"
else
    echo "Warning: Armadillo not found via pkg-config"
    if [ "$(uname)" == "Darwin" ]; then
        echo "Try: brew install armadillo"
    else
        echo "Try: sudo apt-get install libarmadillo-dev"
    fi
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info _skbuild/

# Install build dependencies
echo ""
echo "Installing Python build dependencies..."
pip install -q nanobind 'scikit-build-core[pyproject]' numpy

# Build and install
echo ""
echo "Building extension module..."
pip install -e . -v

# Run tests if pytest is available
if command -v pytest >/dev/null 2>&1; then
    echo ""
    echo "Running tests..."
    pytest fastcpd/tests/ -v
else
    echo ""
    echo "pytest not found. Skipping tests."
    echo "Install with: pip install pytest"
fi

echo ""
echo "========================================="
echo "Build complete!"
echo "========================================="
echo ""
echo "Test the installation:"
echo "  python -c 'from fastcpd.segmentation import mean; print(\"Success!\")'"
