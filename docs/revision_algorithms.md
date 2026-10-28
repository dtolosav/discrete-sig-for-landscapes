# Revision algorithms and verification boundaries

## Functional k-medians

`src/revision_clustering.py` alternates nearest-centre assignment with pointwise
functional medians. Distances are sums of integrals of absolute differences over
retained levels. Each median minimises this objective for a fixed assignment.
Nearest-centre reassignment cannot increase the objective. The implementation
checks this descent with relative numerical allowance `1e-10 * max(1, old_cost)`;
the allowance is a diagnostic tolerance, not a stopping rule.

Initial centres are distinct observations drawn without replacement from a fixed
seed. Initial distance ties choose the lowest centre index. Later exact ties keep
the previous cluster if possible. Empty clusters retain their previous centre,
which preserves the descent argument. As with degenerate k-means inputs, fewer
than k occupied clusters are possible and are explicitly reported. Even-sized
clusters use the midpoint of the two middle pointwise values. Stop on unchanged
assignments, or flag an iteration limit at 300. At every returned step the labels
refer to the returned centres and the objective is recomputed for those labels.

Median breakpoints include pairwise interior crossings between linear pieces.
The pairwise search is vectorised over grid intervals. One- and two-member
clusters need no crossings: their median is the pointwise mean. The reference
algorithm is exact for continuous PWL inputs up to floating-point arithmetic,
but its full-cohort cost has not been established by small tests.

## Restarts

The seed matrix is recorded explicitly: 100 outer seeds from RandomState(42),
and ten start seeds from each outer seed. Each vector start runs sklearn KMeans
with k-means++, Lloyd's algorithm, n_init=1, max_iter=300, tol=1e-4. Each functional
start uses the algorithm above. Each outer run retains the lowest objective
(first start on an objective tie). ARI/NMI labels never enter selection.

Every start's seed, labels, objective, iterations and occupied-cluster count are
saved, plus L1 descent histories and winning centres. Iteration limits are flagged.
Different objectives and initialisation schemes are disclosed; equal numbers of
starts are not a guarantee of equal optimisation difficulty.

## Exact constant signature coordinates

Degree-one discrete-signature coordinates telescope to endpoint displacement.
The FRUITS wrapper identifies these coordinates by their word labels and sets
them from the endpoints, preventing accumulated roundoff from entering scaling.

For a repeated letter j at tensor degree q, the coordinate is exactly
`displacement_j**q / q!`. The Chen wrapper imposes this algebraic identity before
standardisation. Otherwise roundoff in a theoretically zero loop coordinate can
be amplified into a spurious feature. This uses no fitted numerical threshold and
does not discard genuine small varying coordinates. Pure-time coordinates on a
common interval are also fixed constants. Old pilot retained-column counts must
not be reused after this correction; raw dimensions do not change.

## Integrated landscapes

Integrating a PWL landscape produces a piecewise-quadratic path. Taking a Chen
signature of straight segments between its integrated samples is generally an
approximation. `src/integrated_chen.py` instead integrates the signature ODE on
each interval. In the normalised interval variable u in [0,1], the derivative is
`a+b*u`, with `a=h*lambda_left`, `b=h*(lambda_right-lambda_left)`.

The degree-q tensor polynomial satisfies

`dS_q(u)/du = S_(q-1)(u) tensor (a+b*u)`.

Its constant term is the accumulated signature at the interval's start. Each
coefficient of u^p in S_(q-1) contributes its tensor product with a divided by
p+1 to degree p+1, and with b divided by p+2 to degree p+2. Compute increasing
tensor levels; evaluation at u=1 gives the state for the next interval. Level q
has polynomial degree at most 2q. The empty-word coordinate is one and excluded
from returned features. Numba compiles the explicit loops without fastmath.

This computes the integrated-path signature without a discretisation parameter,
up to floating-point error, provided the input grid contains its derivative's
breakpoints. Tests check an analytic quadratic path, one-dimensional exponential
signature, subdivision invariance, and convergence of an independent pySigLib
polygonal approximation on a genuinely curved example. This is a corrected
evaluation of the intended integrated-path comparator; historical numerical
results used a different approximation and require replacement.

## Independent mathematical audit (2026-10-28)

A separate adversarial reviewer checked the recurrence, interval carry-over,
word ordering, repeated-letter identity and functional objective descent. An
independent dictionary-of-words implementation using exact rational arithmetic
agreed with the integrated recurrence through depth four to 1.39e-17 on a
multi-interval noncollinear example. A separate expansion of
`exp(e1) exp(e2) exp(e1)` agreed with pySigLib's ordering through depth four.
Twenty additional synthetic functional clustering runs with differing knot
locations converged without objective increases and agreed with independent
objective recomputation to 2.22e-16. These are checks, not formal verification.

The audit identified a degenerate convergence-test fixture: its integrated path
was straight, so polygonal approximation was already exact and strict error
decrease was inappropriate. The checked-in test now uses a noncollinear example.
No blocking mathematical defect remained. Floating-point error, local rather
than global clustering optima, possible empty clusters, and full-grid runtime
remain relevant limitations.
