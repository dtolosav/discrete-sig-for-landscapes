# Setup Guide

This directory contains setup utilities for the *Discrete Signatures in Topological Data Analysis* project.

The repository is primarily tested on Linux (Ubuntu). Some dependencies (`fruits`, `pysiglib`) include compiled extensions and may require minor adjustments depending on the host system and compiler configuration.

---

# Requirements

- Linux
- Python 3.8+
- C/C++ build tools (`gcc`, `g++`)
- `git`

---

# Installation

## Option 1: Automated Setup

```bash
./setup.sh
```

The setup script is intended to:
- verify the Python installation
- create a virtual environment
- install project dependencies
- create standard data/result directories

If the script fails, follow the manual installation instructions below.

---

## Option 2: Conda Environment

```bash
conda env create -f environment.yml
conda activate signature-tensors-tda
```

---

## Option 3: Manual Pip Installation

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If using `pysiglib` manually:

```bash
# Disable CUDA/GPU support
export CUSIG=0

pip install pysiglib
```

---

# Verification

Verify the installation using:

```bash
make verify
```

or run:

```bash
python scripts/1_process_landscapes.py --help
```

---

# Optional Make Commands

Convenience commands are available through the Makefile:

```bash
make help
make setup
make verify
make clean
```

---

# Common Issues

## Missing Build Tools

Ubuntu/Debian:

```bash
sudo apt install build-essential python3-dev
```

Fedora:

```bash
sudo dnf install gcc gcc-c++ python3-devel
```

---

## Permission Denied

```bash
chmod +x setup.sh
./setup.sh
```

---

# Next Steps

1. Activate the environment:

```bash
source venv/bin/activate
```

2. Download persistence landscapes and preprocess data.

See the main `README.md` for:
- dataset sources
- preprocessing instructions
- clustering experiments
- reproducibility details

---

# Getting Help

- Consult the main `README.md`
- Use script help messages:

```bash
python scripts/<script>.py --help
```

- See `docs/setup.md` for additional troubleshooting information
