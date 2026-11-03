# Discrete signature tensors for persistence landscapes

Code for the paper by Vincenzo Galgano, Heather A. Harrington and Daniel Tolosa.
The revised application compares six landscape representations on 363 knotted
protein chains. **x1** (C-alpha atoms) is the main comparison; **x6** (five added
points between consecutive atoms) is the appendix comparison.

## Install

Use Ubuntu/Linux, or Ubuntu under **WSL2 on Windows**. On Windows, install WSL
from PowerShell with `wsl --install -d Ubuntu`, restart if requested, then open
Ubuntu. All remaining commands run in that Linux terminal. Keep the clone and
environment under your Linux home directory (`~/`), rather than `/mnt/c/`.

Install [Miniforge](https://github.com/conda-forge/miniforge#install) and the OS
tools `git`, `gcc`, `g++` and `make` (on Ubuntu: `sudo apt install git build-essential`).
Then:

```bash
git clone --recurse-submodules https://github.com/dtolosav/discrete-sig-for-landscapes.git
cd discrete-sig-for-landscapes
conda create -n discrete-sig python=3.11 pip
conda activate discrete-sig
bash setup.sh
python -m pytest -q tests
```

`setup.sh` installs the tested package versions, CPU-only PyTorch and the pinned
FRUITS submodule. No GPU or notebook is needed. The tested platform is Ubuntu
24.04 x86-64, including WSL2. The paper release is **v1.0.0**; use
`git checkout v1.0.0` followed by `git submodule update --init --recursive` to
work with that fixed version.

## Regenerate the paper's tables and figures

This uses the saved campaign records and requires no new clustering or raw-data
download:

```bash
python scripts/fetch_results.py
python scripts/report_revision.py --paper-bundle results/paper-results-v1.0.0
```

Outputs appear in `results/paper/`: four tables (CSV and LaTeX rows), three
figures (PDF/PNG), and numerical metadata. Scores are recalculated from saved
assignments; figures retain the paper's selection rules. The bundle download
is checksummed and contains original provenance records, not just final scores.

## Run a small example

```bash
python scripts/fetch_inputs.py
python scripts/run_revision.py --scope pilot --output-dir results/demo
python scripts/report_revision.py --run-dir results/demo
```

The example runs all six methods on 18 proteins with two ten-start repetitions.
It tests the workflow; its smaller critical grid and scores are not paper results.
Allow a few minutes after installation, including first-use compilation.

## Recompute the experiments

```bash
python scripts/run_revision.py --config configs/x1.json --scope full --output-dir results/x1
python scripts/run_revision.py --config configs/x6.json --scope full --output-dir results/x6
```

Run these sequentially. Repeat the same command to resume; completed proteins
and clustering runs are reused. Full runs are expensive: DLFM vectorisation
took about 2.75 hours for x1 on the reference machine, and functional L1 requires
much longer. Use the saved-results route above for a quick examination of the paper.
See [reproduction details](docs/reproduce.md) for timing commands, conventions,
resource guidance, validation and the mapping from files to manuscript results.

## Where to look

- `src/revision_features.py`: landscape evaluation, signatures and standardisation.
- `src/integrated_chen.py`: polynomial integration of integrated-landscape signatures.
- `src/functional_l1.py`, `src/revision_clustering.py`: distances, medians and clustering.
- `src/revision_experiment.py`: sequential execution and checkpoints.
- `src/paper_reporting.py`, `src/paper_plots.py`: paper outputs.
- `configs/x1.json`, `configs/x6.json`: exact reported settings; `tests/`: focused checks.

Inputs come from Benjamin et al., *Homology of homologous knotted proteins*;
see [attribution](data/external/attribution.md). Reproduction starts from their
published landscapes, not from a new persistence computation on PDB coordinates.
The earlier implementation is preserved in Git history and the
`pre-vjom-revision` tag. It is not the implementation of the revised results.

Please cite [the paper](https://doi.org/10.48550/arXiv.2505.02800) and the code
release used. Citation metadata are in `CITATION.cff`. Code licence: MIT.
