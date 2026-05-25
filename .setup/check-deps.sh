#!/bin/bash

# Quick verification script to check if the installation is correct

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Installation Verification ===${NC}"
echo ""

# Check Python
echo -n "Checking Python... "
if python3 --version &> /dev/null; then
  echo -e "${GREEN}✓$(python3 --version)${NC}"
else
  echo -e "${RED}✗ Python 3 not found${NC}"
  exit 1
fi

# Check venv
echo -n "Checking virtual environment... "
if [ -z "$VIRTUAL_ENV" ]; then
  echo -e "${YELLOW}⚠ Not in a virtual environment${NC}"
  echo "   Activate with: source venv/bin/activate"
  VENV_ACTIVE=0
else
  echo -e "${GREEN}✓ Active: $VIRTUAL_ENV${NC}"
  VENV_ACTIVE=1
fi

echo ""
echo "Checking Python packages:"
echo ""

check_package() {
  local pkg=$1
  local import_name=${2:-$1}
  echo -n "  $pkg ... "
  if python3 -c "import $import_name" 2>/dev/null; then
    VERSION=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'installed'))" 2>/dev/null || echo "installed")
    echo -e "${GREEN}✓ $VERSION${NC}"
  else
    echo -e "${RED}✗ Not installed${NC}"
  fi
}

check_package "numpy"
check_package "scipy"
check_package "pandas"
check_package "sklearn" "sklearn"
check_package "matplotlib"
check_package "seaborn"
check_package "tqdm"
check_package "jupyter"
check_package "pytest"

echo ""
echo "Checking specialized packages:"
echo ""

echo -n "  fruits ... "
if python3 -c "import fruits" 2>/dev/null; then
  echo -e "${GREEN}✓ installed${NC}"
else
  echo -e "${YELLOW}⚠ Not installed (Linux/WSL/Docker required)${NC}"
fi

echo -n "  pysiglib ... "
if python3 -c "import pysiglib" 2>/dev/null; then
  echo -e "${GREEN}✓ installed${NC}"
else
  echo -e "${YELLOW}⚠ Not installed (Linux/WSL/Docker required)${NC}"
fi

echo ""
echo "Checking directories:"
echo ""

for dir in data data/external data/processed results notebooks; do
  echo -n "  $dir/ ... "
  if [ -d "$dir" ]; then
    echo -e "${GREEN}✓ exists${NC}"
  else
    echo -e "${YELLOW}⚠ missing${NC}"
  fi
done

echo ""
echo -e "${GREEN}=== Verification Complete ===${NC}"
echo ""

if [ $VENV_ACTIVE -eq 0 ]; then
  echo "Note: Activate the virtual environment to test specialized packages:"
  echo "  source venv/bin/activate"
  echo "  python .setup/check-deps.sh"
fi
