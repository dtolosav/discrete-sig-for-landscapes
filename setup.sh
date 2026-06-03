#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Discrete Signatures in TDA - Linux Setup ===${NC}"
echo ""

# 1. Check OS
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
  echo -e "${RED}Error: This setup script is designed for Linux only.${NC}"
  echo "For macOS, use: pip install -r requirements.txt"
  echo "For Windows, use WSL2 or Docker."
  exit 1
fi

echo -e "${YELLOW}Step 1: Checking system dependencies...${NC}"

# Check for Python 3
if ! command -v python3 &> /dev/null; then
  echo -e "${RED}Error: Python 3 not found. Install with:${NC}"
  echo "  sudo apt-get update && sudo apt-get install python3 python3-venv python3-dev"
  exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check for gcc/g++ (needed for pySigLib)
if ! command -v gcc &> /dev/null || ! command -v g++ &> /dev/null; then
  echo -e "${YELLOW}⚠ gcc/g++ not found (required for pySigLib compilation)${NC}"
  echo "Installing build essentials..."
  sudo apt-get update
  sudo apt-get install -y build-essential python3-dev
fi
echo -e "${GREEN}✓ Build tools available${NC}"

# Check for git (optional but useful)
if ! command -v git &> /dev/null; then
  echo -e "${YELLOW}⚠ git not found (recommended)${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Creating virtual environment...${NC}"

VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
  read -p "Virtual environment '$VENV_DIR' already exists. Use it? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
else
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

echo ""
echo -e "${YELLOW}Step 3: Upgrading pip and build tools...${NC}"
pip install --upgrade pip setuptools wheel

echo ""
echo -e "${YELLOW}Step 4: Installing Python dependencies...${NC}"
git submodule update --init --recursive
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo ""
echo -e "${YELLOW}Step 5: Creating data directories...${NC}"
mkdir -p data/external
mkdir -p data/processed
mkdir -p results
echo -e "${GREEN}✓ Directory structure created${NC}"

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source $VENV_DIR/bin/activate"
echo "2. Download landscapes from Benjamin et al. repository (see README.md)"
echo "3. Process landscapes: python scripts/1_process_lan_script.py --directory <path>"
echo "4. Run signature computation: python scripts/2a_compute_discrete_sigs.py ..."
echo ""
echo "For more details, see README.md or run: python scripts/<script_name>.py --help"
echo ""
