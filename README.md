# Discrete Signature Tensors for Persistence Landscapes

This repository contains the code accompanying the paper:

> **Discrete signature tensors for persistence landscapes**
> Vincenzo Galgano, Heather A. Harrington, Daniel Tolosa
> [arXiv:2505.02800](https://arxiv.org/abs/2505.02800)

In this work, we introduce the **Discrete Landscape Feature Map (DLFM)**, a novel vectorization method combining **persistence landscapes** from topological data analysis with **discrete signature tensors**, a tool from algebraic and time-series analysis. The DLFM captures topological features of data with enhanced discriminative power and stability, and we apply it to biological datasets of knotted proteins.

This repository implements the DLFM pipeline and applies it to protein classification tasks, building on the dataset and ideas in the paper:

> **Homology of homologous knotted proteins**
> Benjamin et al., [arXiv:2201.07709](https://arxiv.org/abs/2201.07709)

We use the software FRUITS for the discrete signature computations:

> **FRUITS: Feature Extraction Using Iterated Sums for Time Series Classification**
> Joscha Diehl, Richard Krieg, [arXiv:2311.14549](https://arxiv.org/abs/2311.14549)
> *GitHub Repository*. https://github.com/irkri/fruits

## Overview

We provide a pipeline for applying the DLFM to protein structure classification. The key steps include:

1. **Reading persistence landscapes** (in `.lan` format, provided by authors of [arXiv:2201.07709](https://arxiv.org/abs/2201.07709))
2. **Computing discrete signature tensors** from the critical points of landscapes
3. **Clustering and analysis** using K-means, PCA, and statistical testing to assess classification and knot-depth correlation

## Repository Structure

* `notebooks/`

  * `1_compute_discrete_sig.ipynb` – loads `.lan` files and computes discrete signatures
  * `2_k_means.ipynb` – clustering of signature data
  * `3_statistical_significance_k_means_assay.ipynb` – permutation testing for statistical significance
  * `4_knot_depth_analysis.ipynb` – signature correlation with knot depth
  * `5_PCA.ipynb` – PCA visualizations of discrete signatures

## Dependencies

This project relies on:

* [`FRUITS`](https://github.com/dtolosa/fruits) (Feature Extraction Using Iterated Sums) – included as a submodule
* Python packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, `tqdm`

### Using the FRUITS Submodule

Make sure to initialize the submodule:

```bash
git clone --recurse-submodules https://github.com/dtolosav/discrete-sig-for-landscapes
```

If you already cloned the repo without the `--recurse-submodules` flag, you can initialize the submodule with:

```bash
git submodule update --init --recursive
```

## Data

We use protein landscape files (`.lan`) provided by the authors of [arXiv:2201.07709](https://arxiv.org/abs/2201.07709). These are not included in the repository. Please contact the original authors or refer to their dataset access instructions.

## Citation

If you use this code, please cite:

Tolosa, D., Galgano, V., Harrington, H. A. (2025). *Discrete signature tensors for persistence landscapes*. arXiv:2505.02800. [https://arxiv.org/abs/2505.02800](https://arxiv.org/abs/2505.02800)

## Contact

Questions or comments? Contact:

**Daniel Tolosa**
Presidential Postdoctoral Fellow
Arizona State University
[dtolosav@asu.edu](mailto:dtolosav@asu.edu)
