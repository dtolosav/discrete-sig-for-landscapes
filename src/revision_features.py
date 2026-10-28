"""Fixed-grid feature primitives for the revision (arrays: sample, level, time)."""
import numpy as np
from math import factorial
from functools import lru_cache
from sklearn.preprocessing import StandardScaler


def critical_grid(landscapes, support):
    points = [0.0, float(support)]
    points.extend(x for levels in landscapes for level in levels for x, _ in level)
    grid = np.unique(points)
    if support <= 0 or grid[0] < 0 or grid[-1] > support:
        raise ValueError("Landscape outside common [0,L]")
    return grid


def evaluate(landscapes, grid):
    grid = np.asarray(grid, dtype=float)
    if len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("Grid must be strictly increasing")
    d = len(landscapes[0])
    if any(len(levels) != d for levels in landscapes):
        raise ValueError("Zero-extend levels before evaluation")
    out = np.zeros((len(landscapes), d, len(grid)))
    for i, levels in enumerate(landscapes):
        for r, points in enumerate(levels):
            if points:
                x, y = np.asarray(points, dtype=float).T
                if not np.all(np.diff(x) > 0):
                    raise ValueError("Landscape knots must be strictly increasing")
                out[i, r] = np.interp(grid, x, y, left=0, right=0)
    return out


def reconstruction_errors(landscapes, grid):
    """Exact sup error for continuous PWL levels: check knots of both curves."""
    sampled = evaluate(landscapes, grid)
    errors = np.zeros(len(landscapes))
    for i, levels in enumerate(landscapes):
        for r, points in enumerate(levels):
            if points:
                x, y = np.asarray(points).T
                if x[0] < grid[0] or x[-1] > grid[-1]:
                    raise ValueError("Grid must cover landscape support")
                error = np.max(np.abs(y - np.interp(x, grid, sampled[i, r])))
                errors[i] = max(errors[i], error)
    return errors


def chen_features(values, grid, weight, time_augmented=False):
    from pysiglib import sig
    grid = np.asarray(grid, dtype=float)
    if values.ndim != 3 or values.shape[2] != len(grid):
        raise ValueError("Expected sample-by-level-by-time values")
    paths = values.transpose(0, 2, 1)
    if time_augmented:
        time = np.broadcast_to(grid[None, :, None], (len(paths), len(grid), 1))
        paths = np.concatenate((time, paths), axis=2)
    features = np.stack([np.asarray(sig(np.array(p, order="C", copy=True), weight)) for p in paths])
    # The repeated-letter identity is exact: S_{j...j} = displacement_j^q/q!.
    # Enforce known zeros/constants before standardisation, so roundoff in
    # closed-path coordinates is not amplified into apparent features.
    dim = paths.shape[2]
    displacement = paths[:, -1] - paths[:, 0]
    offset = 0
    for q in range(1, weight + 1):
        for j in range(dim):
            index = offset + j * sum(dim**r for r in range(q))
            features[:, index] = displacement[:, j]**q / factorial(q)
        offset += dim**q
    return features


@lru_cache(maxsize=1)
def _discrete_plan(dim, weight):
    """Reuse only the dimension/weight plan in the sequential driver.

    INC and END learn no data-dependent parameters. fit/transform below still
    replace the per-input cache; retaining one plan bounds resident memory.
    """
    import fruits
    fruit = fruits.Fruit("revision")
    fruit.add(fruits.preparation.transform.INC)
    iss = fruits.ISS(fruits.words.of_weight(weight, dim=dim),
                     mode=fruits.ISSMode.EXTENDED)
    fruit.add(iss)
    fruit.add(fruits.sieving.END)
    columns = {iss.label(i): i for i in range(iss.n_iterated_sums())}
    degree_one = []
    for j in range(dim):
        letter = str(j + 1) if j < 9 else f"({j + 1})"
        degree_one.append(columns[f"[{letter}]"])
    return fruit, degree_one


def discrete_features(values, weight):
    fruit, degree_one = _discrete_plan(values.shape[1], weight)
    fruit.fit(values)
    features = fruit.transform(values)
    # The degree-one sums telescope. FRUITS coordinates are not weight-sorted.
    features[:, degree_one] = values[:, :, -1] - values[:, :, 0]
    return features


def standardise(features):
    """Return values, scaler and original-column mask for persisted reuse."""
    features = np.asarray(features, dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite")
    keep = np.any(features != features[0], axis=0)
    if not keep.any():
        raise ValueError("No nonconstant features for clustering")
    scaler = StandardScaler()
    return scaler.fit_transform(features[:, keep]), scaler, keep


def seed_manifest(base_seed, runs, initialisations):
    run_seeds = np.random.RandomState(base_seed).randint(0, 2**31 - 1, runs)
    starts = [np.random.RandomState(int(s)).randint(0, 2**31 - 1, initialisations)
              for s in run_seeds]
    return run_seeds, np.asarray(starts)
