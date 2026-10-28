# Executable revision protocol

This document consolidates D-2026-007 and D-2026-009 through D-2026-018.
It specifies implementation details without changing the approved comparisons.
The active configuration is `configs/revision.json`; actual executions bind
inputs, settings, code, dependency versions and seeds in `manifest.json`.

## Inputs and representations

Use the 363 named x6 chains, sorted by chain identifier; exclude the 150 `Other`
chains. Check every retained file and the classification CSV against the saved
reconstruction audit. Retain 15 landscape levels, zero-extending absent levels.
Remove consecutive identical knots in memory only. Check stored knot depth
against N-tail times C-tail divided by squared chain length, allowing 1e-9 for
the rounded CSV values. The common support is [0,20.951].

The primary signature grid is the union of critical abscissae through level 15
of the analysed cohort, including both support endpoints. It has 107564 points
on the full cohort. All signature methods evaluate on this same grid. Retain
this grid for level and signature-weight ablations: rebuilding it when reducing
levels would also change the discrete increments and confound a one-factor
comparison. This is the implementation of the approved one-factor requirement.

Primary methods:

- DLFM through weight 3, with the pinned FRUITS increment/extended-sums pipeline.
- Ordinary Chen signatures through degree 3.
- Time-augmented Chen signatures of `(t, lambda(t))`, through degree 3, time
  multiplier 1, without per-protein time normalisation.
- Chen signatures through degree 3 of integrated landscapes, using polynomial
  integration of the piecewise-quadratic path up to floating-point error.
- Flattened landscape values on 1024 uniformly spaced points, including endpoints.
- Functional L1 k-medians through level 15, using exact PWL integrals and genuine
  pointwise medians up to floating-point error.

For every vector representation, remove constant columns and fit one
coordinatewise StandardScaler on the analysed cohort; reuse it across starts
and runs. Save the raw/scaled features, mask, means and scales. Enforce known
endpoint identities before standardisation to avoid amplifying roundoff. This
geometry is standardised Euclidean geometry, not Bombieri--Weyl geometry.
Functional L1 uses the unstandardised landscape functions.

The full-cohort reconstruction check supports retaining the already approved
1024-point primary baseline: the maximum relative integrated absolute error was
0.05830% of retained landscape mass, with maximum sup error about 0.010240 in
filtration units. This assessment uses no class-agreement scores and introduces
no fitted adequacy threshold. It does not certify preservation of distances
after standardisation or of clusters. Both approved resolution checks remain
mandatory, and their outcomes must be reported rather than used to select the
most favourable baseline.

## Configurations and optimisation

The full suite has 24 configurations: six primary; six each at 5 and 10 levels;
four signature methods at weight/depth 2; and the flattened method at 512 and
2048 points. Only the named factor changes. The signature grid remains fixed.

Every configuration uses nine clusters, 100 runs and ten starts per run. The
same explicit seed matrix, generated from base seed 42, is used throughout.
Each run selects the first minimum of its own objective across its ten starts;
class labels never select a start or update centres.

Vector clustering uses sklearn KMeans with k-means++, Lloyd's algorithm,
`n_init=1` for each explicit start, `max_iter=300` and `tol=1e-4`. Functional
k-medians starts from distinct observed functions sampled without replacement,
stops on unchanged assignments, and has the same iteration cap. Initially ties
choose the lowest index; later exact ties retain the previous cluster. Empty
centres are retained. Even-sized clusters use the midpoint of the middle two
values. Save all starts, objectives, assignments, limits and occupied counts,
plus the winning centres. Investigate iteration-limit flags before reporting
the experiment as validated. No global optimum is claimed.

Report mean and standard deviation (`ddof=0`, over the specified collection of
runs) of ARI and arithmetic-normalised NMI. These describe initialisation
variability, not population uncertainty. Report all ablations. Do not restore
historical permutation p-values or centroid-separation significance claims.

## Timing and execution

Run sequentially with OMP, OpenBLAS, MKL and Numba thread counts fixed at one.
Record library versions, thread-pool metadata and machine identity. Process
one sampled protein at a time; assemble only the much smaller feature matrix
before standardisation. The complete landscape array would occupy 4.69 GB.

Measure loading/level selection/grid construction/evaluation as preprocessing;
signature computation or flattening plus standardisation as representation;
and all ten starts with objective-based selection as clustering. Save a separate
tiny warmup/setup time for each vector configuration: core tables show warm
encoding, not cold import/JIT latency. Also report the total including this
setup, so cached FRUITS coordinate-plan construction is explicitly charged.
The sequential driver retains only one dimension/weight plan at a time; INC
and END learn no data-dependent parameters. No speed claim should ignore this
timing distinction.

Shared input loading is measured once and charged to each standalone method
total. Signature methods also receive the shared critical-grid construction
cost. Other evaluation costs are method-specific. Standalone totals are not
the actual wall time of a campaign that reuses shared work. Exclude upstream
protein/PH construction, hashing, output IO, plotting and scoring from core
times. On resume, retain completed-stage timings and disclose that repeated
loading/import/warmup overhead is outside these core totals.

The driver checkpoints each encoded protein and each completed ten-start run.
Changed inputs, settings, code, seeds or dependency metadata require a new
output directory. An exact source snapshot preserves the uncommitted code used
for each run. Completed artifacts are hash-checked when reused. Figures
and tables are reconstructed from saved outputs, not from rerun clustering.

## Descriptive depth analysis

Use standardised primary DLFM Euclidean distances and the aligned stored knot
depths. Retain only unordered pairs `i < j`. Report the Spearman coefficient
without its ordinary independence-based p-value, both overall and within/between
sequence classes, with per-class descriptive summaries where defined. Do not
claim prediction, generalisation to new classes, or an independent biological
effect. Prediction remains deferred.

## Validation and full execution

The driver's pilot uses two lexicographically first chains per class (18 total),
the subset's common critical grid, the same full-cohort support, and two outer
runs with ten starts. This reduced scope is explicitly marked in every manifest
and plot. All 24 configurations must complete before the driver is marked ready.
The full cohort retains the approved 100-by-10 budget.

From the verified environment:

```sh
python -m pytest -q tests
python scripts/run_revision.py --scope pilot --output-dir ../results/revision-driver-pilot/2026-10-28-v2
python scripts/report_revision.py --run-dir ../results/revision-driver-pilot/2026-10-28-v2
python scripts/run_revision.py --scope full --prepare-only --output-dir ../results/revision-full/2026-10-28
```

After driver validation, use the same full command without `--prepare-only`.
Repeating it resumes completed work. `--only primary-dlfm` (or any exact
configuration ID from the manifest) runs only that configuration; the manifest
still lists the whole required suite. The report command refuses an incomplete
suite. None of these outputs becomes a manuscript result merely by existing;
inspect diagnostics and complete manuscript/response integration separately.
