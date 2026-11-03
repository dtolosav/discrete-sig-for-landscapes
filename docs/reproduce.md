# Reproducing the revised application

Start with the [README](../README.md). Commands below run from the repository
root, in the activated Python 3.11 environment. No files from the manuscript's
private working directory are required.

## Inputs and settings

`scripts/fetch_inputs.py` retrieves the pinned upstream landscape data and
verifies every degree-one file. The two cohorts contain about 103 MiB (x1) and
26 MiB (x6) of landscape files; Git metadata and other upstream files add to the
download. Both contain 513 chains, of which the same 363 belong to nine named
classes. The corrected classification file and all checksums are versioned.

| Setting | x1 main text | x6 appendix |
|---|---|---|
| Configuration | `configs/x1.json` | `configs/x6.json` |
| Support endpoint | 23.0931 | 20.951 |
| Primary landscape levels | 15 | 15 |
| Signature weight/degree | 3 | 3 |
| Flattened baseline points | 1024, uniform | 1024, uniform |
| Primary methods | 6 | 6 |
| Vector agreement runs | 100 × 10 starts | 100 × 10 starts |
| Functional L1 agreement runs | 10 × 10 starts | 100 × 10 starts |
| Additional comparisons | none | 17, explicitly specified in the configuration |

Signature grids contain the reference cohort's retained critical points and
support endpoints. Level ablations keep the primary fifteen-level grid fixed.
The runner processes one protein at a time, then standardises each complete
feature matrix once across proteins. It removes exactly constant columns,
without a fitted small-value threshold. Time augmentation uses the original
filtration parameter without rescaling.

Discrete first-order coordinates and repeated-letter ordinary/time-augmented
Chen coordinates are set from their exact endpoint identities before scaling.
Integrated-path signatures use polynomial integration of the signature ODE;
see `src/integrated_chen.py` and its analytic tests. Functional L1 uses exact
piecewise-linear absolute integrals and pointwise medians, up to floating-point
arithmetic. It does not standardise the landscape functions.

## Execution and resumption

```bash
python scripts/run_revision.py --config configs/x1.json --scope full --output-dir results/x1
python scripts/report_revision.py --run-dir results/x1
python scripts/run_revision.py --config configs/x6.json --scope full --output-dir results/x6
python scripts/report_revision.py --run-dir results/x6
```

Use `--prepare-only` to validate inputs and save a manifest before computing.
Use `--only primary-dlfm` (or other IDs in the manifest) to run selected methods;
reporting a whole campaign requires all its selected configurations to finish.
Repeating a command resumes completed proteins and runs. Changed numerical
source, environment, inputs or settings require a new output directory.
`status.json` lists completed configurations; each method has `run-000.json`,
etc., and receives `summary.json` when complete. No background scheduler is used.

The seed matrix uses `RandomState(42)` to draw 100 outer seeds, with ten start
seeds drawn from each outer seed. Shorter runs use prefixes of this matrix.
Each start uses nine clusters. Vector clustering uses Lloyd/k-means++, a
300-iteration cap and tolerance 1e-4. Functional k-medians starts from distinct
observations, retains a previous assignment on exact ties where possible,
otherwise chooses the lowest index, and retains centres of empty clusters.
Its even-sized median averages the middle two values; it stops at unchanged
assignments or 300 iterations. Both select the first minimum-objective start.
Sequence labels never enter this selection. NMI uses arithmetic normalisation;
reported standard deviations use `ddof=0`.

## Timing

Run benchmarks alone, sequentially, with the runner's one-thread settings:

```bash
python scripts/run_revision.py --config configs/x1.json --scope timing --output-dir results/timing --only primary-dlfm primary-chen primary-time_chen primary-integrated_chen primary-flattened
```

This computes fresh full-cohort vectors and ten ten-start clustering runs.
The primary functional L1 run already supplies ten timed repetitions.
The paper's vectorisation times include encoding and standardisation; they
exclude loading, grid construction, evaluation on that grid, warmup and file
I/O. Clustering times include all ten starts and exclude scoring and I/O.
The table reports the median over ten runs. Original hardware, package versions
and stage timings are retained in the bundle manifests. Hardware-dependent
times will differ on a new machine; do not substitute them for the original
table without identifying them as new measurements.

## Paper outputs and checks

```bash
python scripts/fetch_results.py
python scripts/report_revision.py --paper-bundle results/paper-results-v1.0.0 --output-dir results/paper
python scripts/audit_paper_inputs.py --paper-bundle results/paper-results-v1.0.0 --output-dir results/input-checks
python scripts/check_reference_features.py --paper-bundle results/paper-results-v1.0.0
```

The first two commands need only the ~93 MiB result archive. The third also
needs the raw landscapes; it recomputes uniform-grid approximation errors and
checks the per-chain level-11 areas underlying the appendix interpretation.
The fourth recomputes two proteins per primary vector method in each regime
on the full paper grids, checking raw and standardised coordinates at
`rtol=atol=1e-12`; allow several minutes.

| Generated file | Paper content |
|---|---|
| `x1_summary_rows.csv/.tex` | Main method comparison, coordinates and timing |
| `x1_depth_rows.csv/.tex` | Unique-pair knot-depth associations |
| `x6_comparison_rows.csv/.tex` | Appendix primary comparison |
| `x6_sensitivity_rows.csv/.tex` | Appendix parameter comparisons |
| `x1_cluster_composition.pdf` | Median-objective x1 partition and all-run histogram |
| `x1_pca_views.pdf` | The same PCA projection coloured two ways |
| `x6_clustering_families.pdf` | Objective-selected examples within the two stated families |
| `paper_metadata.json` | Figure selections, depth correlations and within-class observations |

Bundle files are checked against their hashes. Scores are recomputed from saved
assignments; the lowest-objective start is checked independently. Original
manifests remain unchanged. Most feature caches and restart centres are omitted;
reporting does not claim to reverify those omitted files. All original run
assignments, objectives and source records needed for the reported comparisons
are included. Full recomputations produce their own complete checkpoints.

The x1 example is closest to the median objective, with the first tie chosen.
The x6 examples minimise the objective within the retrospective ARI > 0.9 and
ARI < 0.65 groups. All 100 outcomes appear in the histograms. PCA uses the full
standardised DLFM matrix with the full SVD solver; no dimensionality reduction
is applied before clustering. Depth correlations count unordered pairs once
and are descriptive, without independent-pair significance tests.

## Validation and resources

```bash
python -m pytest -q tests
python -m pytest -q fruits/tests
```

The tests cover signatures, functional medians, objective selection, seed
prefixes and checkpoint resumption. The Ubuntu CI job installs into an empty
container and runs the real-data demonstration. These checks and regeneration
of saved results are distinct from rerunning every full campaign. See the
versioned release notes for the actual fresh-install and regression results.

Use at least several GiB of free memory and disk for a full run; do not build a
single dense array containing every protein on the critical grid. DLFM encoding
and functional L1 dominate full-run cost. Keep results on the Linux filesystem
for WSL runs. The saved-results route is the practical starting point for review.
