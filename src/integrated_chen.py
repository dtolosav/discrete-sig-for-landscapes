"""Chen signatures of integrals of piecewise-linear landscape paths.

An integrated landscape is piecewise quadratic, not piecewise linear. On each
interval, use u in [0,1], so its derivative is a+b*u. Solve the signature ODE
dS_q/du = S_{q-1} tensor (a+b*u) exactly by polynomial integration (up to
floating-point arithmetic). Tensor words use the same lexicographic order as
pySigLib. The recurrence is explicit below; Numba only compiles the loops.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _integrated_signature(values, grid, weight):
    n, dim = values.shape
    offsets = np.zeros(weight + 2, dtype=np.int64)
    for q in range(weight + 1):
        offsets[q + 1] = offsets[q] + dim**q
    state = np.zeros(offsets[-1])
    state[0] = 1.
    for interval in range(n - 1):
        width = grid[interval + 1] - grid[interval]
        a = width * values[interval]
        b = width * (values[interval + 1] - values[interval])
        coefficients = np.zeros((2 * weight + 1, offsets[-1]))
        coefficients[0] = state
        for q in range(1, weight + 1):
            previous, current = offsets[q - 1], offsets[q]
            for p in range(2 * (q - 1) + 1):
                for word in range(dim**(q - 1)):
                    coefficient = coefficients[p, previous + word]
                    for letter in range(dim):
                        index = current + word * dim + letter
                        coefficients[p + 1, index] += coefficient * a[letter] / (p + 1)
                        coefficients[p + 2, index] += coefficient * b[letter] / (p + 2)
        state = coefficients.sum(axis=0)
    return state[1:]


def integrated_chen_features(values, grid, weight):
    """Input shape (samples, levels, times); empty-word coordinate excluded."""
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if values.ndim != 3 or values.shape[2] != len(grid) or weight < 1:
        raise ValueError("Invalid sample-by-level-by-time input or weight")
    if len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("Grid must be strictly increasing")
    return np.stack([_integrated_signature(np.ascontiguousarray(path.T), grid, weight)
                     for path in values])
