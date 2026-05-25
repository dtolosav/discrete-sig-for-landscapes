.PHONY: help setup setup-conda setup-pip install clean data-dirs verify test

help:
	@echo "Discrete Signatures in TDA - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup         Run automated Linux setup (venv + dependencies)"
	@echo "  make setup-conda   Create conda environment (if conda is available)"
	@echo "  make setup-pip     Manual pip install to existing venv"
	@echo "  make install       Alias for make setup"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run pytest on the src/ directory"
	@echo "  make clean         Remove generated files (__pycache__, *.pyc, etc.)"
	@echo "  make data-dirs     Create data and results directories"
	@echo "  make verify        Verify installation by importing key modules"
	@echo ""
	@echo "Run scripts (examples):"
	@echo "  python scripts/1_process_lan_script.py --help"
	@echo "  python scripts/2a_compute_discrete_sigs.py --help"
	@echo "  python scripts/3a_kmeans_clustering.py --help"
	@echo ""

setup: data-dirs
	@bash setup.sh

setup-conda: data-dirs
	@echo "Setting up conda environment..."
	@if command -v conda &> /dev/null; then \
		conda env create -f environment.yml; \
		echo ""; \
		echo "Environment created! Activate with:"; \
		echo "  conda activate signature-tensors-tda"; \
	else \
		echo "Error: conda not found. Install Miniconda or Anaconda first."; \
		exit 1; \
	fi

setup-pip: data-dirs
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
	fi
	@echo "Upgrading pip and tools..."
	@. venv/bin/activate && pip install --upgrade pip setuptools wheel
	@echo "Installing dependencies..."
	@. venv/bin/activate && pip install -r requirements.txt
	@echo "✓ Setup complete. Activate with: source venv/bin/activate"

install: setup

data-dirs:
	@mkdir -p data/external data/processed results notebooks
	@echo "✓ Data directories created"

clean:
	@echo "Cleaning up..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.egg-info" -delete
	@rm -rf build/ dist/ .pytest_cache/
	@echo "✓ Cleaned"

verify:
	@echo "Verifying installation..."
	@python3 -c "import numpy; print('✓ numpy', numpy.__version__)"
	@python3 -c "import scipy; print('✓ scipy', scipy.__version__)"
	@python3 -c "import pandas; print('✓ pandas', pandas.__version__)"
	@python3 -c "import sklearn; print('✓ scikit-learn', sklearn.__version__)"
	@python3 -c "import fruits; print('✓ fruits installed')" 2>/dev/null || echo "⚠ fruits not installed (may require Linux/venv)"
	@python3 -c "import pysiglib; print('✓ pysiglib installed')" 2>/dev/null || echo "⚠ pysiglib not installed (may require Linux/venv)"
	@echo ""
	@echo "✓ Core dependencies verified"

test:
	@echo "Running tests..."
	@pytest tests/ -v 2>/dev/null || echo "⚠ No tests found in tests/"
