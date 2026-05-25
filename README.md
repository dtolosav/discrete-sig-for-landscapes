# Discrete Signatures in Topological Data Analysis

This repository contains code and experiments accompanying the paper *Discrete signature tensors for persistence landscapes*. The repository implements the **Discrete Landscape Feature Map (DLFM)**, a vectorisation of persistence landscapes based on discrete signatures of critical-point time series.

We compare DLFM against continuous (Chen) signatures, integrated landscape signatures, and standard landscape vectorisations on a dataset of knotted proteins. The experiments show that DLFM provides substantially improved discriminative power for recovering biologically meaningful structure from persistent homology features.

For mathematical details and theoretical results, see the accompanying paper.

---

## Main Results

On a dataset of knotted proteins, DLFM:

- Outperforms continuous (Chen) signatures and standard landscape vectorisations in clustering tasks
- Achieves high agreement with sequence similarity classes (ARI ≈ 0.96, NMI ≈ 0.88)
- Provides substantially more compact feature vectors than direct landscape vectorisations
- Supports downstream machine learning tasks such as knot-depth prediction

---

## Repository Structure

```text
signature-tensors-tda/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── environment.yml                     # Conda environment definition
├── requirements.txt                    # Pip requirements
├── scripts/
│   ├── 1_process_landscapes.py         # Landscape preprocessing
│   ├── 2a_compute_discrete_sigs.py     # DLFM computation
│   ├── 2b_compute_Chen_sigs.py         # Continuous signature baselines
│   ├── 3a_kmeans_clustering.py         # Clustering on signature features
│   ├── 3b_kmeans_on_flat_landscapes.py # Direct landscape vectorisation baseline
│   └── 3c_kmeans_on_landscapes_L1.py   # L1-distance baseline
├── data/
│   ├── README.md
│   └── external/
│       └── attribution.md
├── results/                            # Computed signatures and clustering outputs
├── docs/
│   └── setup.md                        # Detailed installation/troubleshooting
└── references.bib                      # Bibliography references
```

---

# Quick Start

## 1. Environment Setup

The repository is primarily tested on Linux.

We provide both Conda- and pip-based installation workflows. Some dependencies (`fruits`, `pysiglib`) include compiled extensions and may require WSL or Docker on Windows systems.

### Option A: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate signature-tensors-tda
```

### Option B: Pip + Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option C: Manual Installation

```bash
pip install -U pip setuptools wheel
pip install numpy scipy pandas scikit-learn matplotlib seaborn tqdm jupyter

# CPU-only pySigLib
export CUSIG=0

pip install pysiglib
```

See `docs/setup.md` for additional setup instructions and troubleshooting.

---

## 2. Download & Preprocess Landscapes

Persistence landscapes can be obtained from:

- **Option A (Recommended):** Download preprocessed landscapes from the repository accompanying Benjamin et al. (2023):
  https://github.com/katherine-benjamin/ph-knotted-proteins

- **Option B:** Download raw protein data from KnotProt2.0 and compute landscapes independently:
  https://knotprot.cent.uw.edu.pl/

Place `.lan` files in a directory, then preprocess:

```bash
python scripts/1_process_landscapes.py \
  --directory path/to/landscapes \
  --trunc-level 15 \
  --output-dir data/
```

This generates compressed NumPy arrays ready for signature computation.

---

## 3. Compute Signatures

### Compute DLFM Features (Discrete Signatures)

Requires the `fruits` package.

```bash
python scripts/2a_compute_discrete_sigs.py \
  --input data/landscapes_array_L15_*.npz \
  --weight 3 \
  --csv
```

### Compute Continuous (Chen) Signatures

Requires `pysiglib`.

```bash
python scripts/2b_compute_Chen_sigs.py \
  --input data/landscapes_array_L15_*.npz
```

This computes:
- Chen signatures of landscapes
- Chen signatures of integrated landscapes

---

## 4. Clustering Experiments

### K-means on Signature Features

```bash
python scripts/3a_kmeans_clustering.py \
  --input results/discrete_signatures_weight3_*.npz \
  --n-clusters 9
```

### K-means on Flattened Landscapes

```bash
python scripts/3b_kmeans_on_flat_landscapes.py \
  --input data/processed/landscapes_array_L15_*.npz \
  --n-clusters 9
```

### K-means Using L1 Landscape Distances

```bash
python scripts/3c_kmeans_on_landscapes_L1.py \
  --input path/to/raw/lan/dir \
  --n-clusters 9
```

See script help messages (`--help`) for additional options and parameters.

---

# Dependencies

## Core Requirements

- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Signature Computation

- `fruits` — discrete signature computation
- `pysiglib` — continuous (Chen) signature computation

## Optional

- `jupyter`
- `jupyterlab`

---

# Data

## Source Attribution

All protein data originates from public sources.

- **Protein structures:** Protein Data Bank (PDB)
- **Knotted protein annotations:** KnotProt2.0
- **Persistence landscapes:** Benjamin et al. (2023)
- **Classification labels:** `trefoil_list.csv` from Benjamin et al.

See:
- `data/README.md`
- `data/external/attribution.md`

for complete attribution and licensing information.

---

# Reproducibility

The published experiments use fixed random seeds to ensure reproducibility of clustering results. Minor numerical differences may still arise across platforms or library versions.

---

# Citation

If you use this code, please cite the accompanying paper:

```bibtex
@misc{galgano_discrete_2025,
	title = {Discrete signature tensors for persistence landscapes},
	url = {http://arxiv.org/abs/2505.02800},
	doi = {10.48550/arXiv.2505.02800},
	author = {Galgano, Vincenzo and Harrington, Heather A. and Tolosa, Daniel},
	month = may,
	year = {2026},
}
```

Please also cite the main software dependencies:

- **FRUITS:** Diehl & Krieg — *Feature Extraction Using Iterated Sums*
- **pySigLib:** Shmelev & Salvi — *pysiglib*
- **Benjamin et al. (2023):** *Homology of homologous knotted proteins*

---

# References

See `references.bib` for the complete bibliography.

Key references include:

1. Diehl & Krieg — *FRUITS: Feature Extraction Using Iterated Sums*
2. Benjamin et al. — *Homology of homologous knotted proteins*
3. Bubenik — *Statistical topological data analysis using persistence landscapes*
4. Edelsbrunner & Harer — *Computational Topology: An Introduction*

---

# Troubleshooting

## pySigLib Installation

On Linux, ensure that a C++ compiler is available:

```bash
sudo apt install build-essential
```

## Landscape Processing

- `.lan` files should follow the standard persistence landscape format
- Use `--trunc-level 15` for consistency with the published experiments
- Use `--exclude-other` to remove proteins labeled `"Other"`

---

# Contributing

This repository accompanies a peer-reviewed research paper. After publication, we welcome:

- Bug reports and feature requests via GitHub Issues
- Pull requests implementing improvements or extensions
- Applications to additional datasets and benchmarking studies

---

# License

MIT License — see `LICENSE` for details.

---

**Corresponding Author:** Daniel Tolosa  
**Affiliation:** Arizona State University  
**Contact:** dtolosav@asu.edu
