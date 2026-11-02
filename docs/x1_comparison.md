# Primary x1 comparison, 2 November 2026

Author-approved supplementary comparison: the five primary vector methods only,
100 ten-start runs, 15 landscape levels, weight/degree 3, and 1024 points for
flattened landscapes. No functional L1 or ablations. The x6 campaign is unchanged.

Run from the implementation repository in the verified WSL environment:

```sh
python -u scripts/run_x1_comparison.py --prepare-only --output-dir ../results/x1-comparison/2026-11-02
python -u scripts/run_x1_comparison.py --output-dir ../results/x1-comparison/2026-11-02
```

Repeat the second command to resume. Preparation requires the completed
`../results/x1-diagnostic/2026-11-02/` archive and the x6 campaign manifest.
It validates input hashes, labels, source files and seed alignment, checks a
fresh full-grid DLFM row, and imports the corrected diagnostic features.
Clustering is rerun for all 100 repetitions so the standard runner saves
centres and complete restart records. Source snapshots and hashes accompany
the run. Reports are generated only after all five methods complete and pass
the existing reporting checks.

The support is [0,23.0931], with 112049 critical grid points. These are determined
from x1, using the same rule as x6. No per-protein time scaling is introduced.

Timings are diagnostic only: the x6 computation may run concurrently, and
cached DLFM row times combine evaluation and encoding and omit original setup.
The generated timing columns must not replace the manuscript runtime table.

Progress: `status.json`, individual `primary-*/summary.json` files and the
launcher log. Final verified scores: `reports/comparisons.csv`. Raw data and
results remain outside Git; the runner and these instructions are versioned.
