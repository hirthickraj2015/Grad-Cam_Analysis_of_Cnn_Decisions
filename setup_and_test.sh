#!/bin/bash

# Adaptive Integrated Grad-CAM - Setup and Test Script
# =====================================================
# This script creates a uv environment, installs dependencies, and tests the code

set -e  # Exit on error

echo "========================================="
echo "Adaptive Integrated Grad-CAM Setup"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if uv is installed
echo -e "${BLUE}[1/5] Checking for uv installation...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ uv is not installed!${NC}"
    echo ""
    echo "Please install uv first:"
    echo "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Or visit: https://github.com/astral-sh/uv"
    exit 1
else
    uv_version=$(uv --version)
    echo -e "${GREEN}✓ uv is installed: ${uv_version}${NC}"
fi

# Create virtual environment
echo ""
echo -e "${BLUE}[2/5] Creating uv virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠ .venv already exists, removing it...${NC}"
    rm -rf .venv
fi

uv venv .venv
echo -e "${GREEN}✓ Virtual environment created at .venv${NC}"

# Activate virtual environment
echo ""
echo -e "${BLUE}[3/5] Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo ""
echo -e "${BLUE}[4/5] Installing dependencies from requirements.txt...${NC}"
echo "This may take a few minutes..."
echo ""

uv pip install -r requirements.txt

echo ""
echo -e "${GREEN}✓ All dependencies installed successfully!${NC}"

# Run tests
echo ""
echo -e "${BLUE}[5/5] Running comprehensive tests...${NC}"
echo ""

python quick_test.py

# Check test exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "Your environment is ready to use!"
    echo ""
    echo "To activate the environment in the future, run:"
    echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Read PROJECT_GUIDE.md for detailed instructions"
    echo "  2. Run: python example_single_image.py --image <your_image.jpg>"
    echo "  3. Run: python demo_experiments.py for full experiments"
    echo ""
else
    echo ""
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}✗ TESTS FAILED${NC}"
    echo -e "${RED}=========================================${NC}"
    echo ""
    echo "Please check the error messages above and resolve any issues."
    exit 1
fi
