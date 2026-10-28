# Revision implementation checks

`configs/revision.json` records the approved configuration. Its status tracks
driver validation; `ready` permits full execution. The consolidated
[protocol](revision_protocol.md) and the commands in the README are the current
workflow. The earlier feature/clustering pilots below remain provenance checks.

From the repository in the existing WSL `discrete-sig` environment:

```sh
python -m pytest -q tests
python scripts/revision_pilot.py --output-dir ../results/revision-pilot/2026-10-28
python scripts/revision_clustering_pilot.py --output-dir ../results/revision-clustering-pilot/2026-10-28
python scripts/revision_grid_audit.py --output-dir ../results/revision-grid-audit/2026-10-28
python scripts/revision_feasibility.py --output-dir ../results/revision-feasibility/2026-10-28
```

The pilot scans the named x6 cohort to determine the common support endpoint,
then selects the first three named chains lexicographically. It computes a
critical grid for this subset only. This is an implementation check, not a
representative performance sample or the manuscript's full-cohort grid.

The JSON report records selection, checksums, configuration, reconstruction
errors, feature dimensions, and cold-run timings. The NPZ preserves raw and
standardised features, masks, scaler parameters, and the 100-by-10 seed manifest.
No clustering score or statistical significance is inferred from this pilot.

The clustering pilot selects two chains per named class (18 proteins), explicitly
uses only two outer runs with ten starts each, and tests all six methods. This
reduced budget is for implementation validation; the full protocol remains
100-by-10. Its report saves every start and winning centres, with ARI/NMI solely
as pilot diagnostics. It does not establish comparative scientific performance.
See [algorithm notes](revision_algorithms.md) for objective, stopping, tie,
constant-coordinate and integrated-signature conventions.

The grid audit reads all 363 named x6 inputs but computes no signatures or
clusters. It measures exact PWL reconstruction errors at 512, 1024 and 2048
points: supremum error and integrated absolute error relative to each protein's
total retained landscape mass. These errors do not certify preservation of
standardised-feature distances, barcode information, or clustering assignments.
The loader removes consecutive identical knots, preserving the represented
function and all source files; conflicting heights at one abscissa are rejected
by the revision evaluation code.

The feasibility script computes signatures for only one protein on the full
cohort grid, then runs ten L1 starts on up to eight proteins per class. It saves
warm execution times, peak process RSS and every L1 start, without class-agreement
scores. It also compares continuous signatures against the same protein's own
breakpoint grid. This is a resource check, not a full comparison. Full processing
should use small batches: the complete sampled array alone would occupy 4.69 GB.

The functional L1 reference implementation inserts all interior pairwise
crossings before taking medians. For even clusters it averages the two middle
values. It integrates absolute PWL differences analytically, splitting at sign
crossings. The historical script's placeholder now calls this implementation,
but its old driver does **not** yet implement the approved run protocol.

Still required before full runs:

- full-cohort grid reconstruction assessment and resolution decision;
- memory-bounded full driver, using the bounded feasibility checks;
- level-ablation grid convention, timing instrumentation, and inference scope;
- output/figure generation and final manifest review.

Completed at pilot scale: common clustering runner with explicit ten starts,
stopping/tie/empty-cluster rules; all six methods on 18 proteins; exact polynomial
integration for the integrated-landscape Chen comparator, with independent
mathematical checks. The detailed verification record is in the workspace's
`REVISION_PLAN.md`. A one-protein full-grid check and a 72-protein L1 check also
completed; these are still pilots, not the full 363-protein experiment.

The historical README result summaries and notebook outputs describe x1 results
and are not evidence for the revised x6 comparison. Prediction is deferred under
D-2026-017; the revised depth analysis is descriptive only.
