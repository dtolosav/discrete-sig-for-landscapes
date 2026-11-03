#!/usr/bin/env bash
# Run from the repository root in an activated Python 3.11 environment.
set -eu
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
git submodule update --init --recursive
# CPU wheels first: pySigLib must not pull in the CUDA stack.
python -m pip install 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e ./fruits
python -m pip check
python -c 'import fruits, pysiglib, torch; assert not torch.cuda.is_available(); print("Signature libraries ready (CPU)")'
