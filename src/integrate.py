import numpy as np

def integrate_lan_on_grid(t_local, lam_local, t_grid):
    """
    Evaluate the integrated persistence landscape L(t)=∫_0^t λ(s) ds
    at times t_grid, given λ by its critical points (piecewise linear).

    Parameters
    ----------
    t_local : (m,) array_like
        Strictly increasing critical points, assumed t_local[0] >= 0.
    lam_local : (m, d) array_like
        Landscape values at critical points: lam_local[i, k] = λ_k(t_local[i]).
    t_grid : (n,) array_like
        Sorted query times where to evaluate L. Typically uniform grid on [0, T].

    Returns
    -------
    L_grid : (n, d) np.ndarray
        Integrated landscape evaluated at t_grid: L_grid[j, k] = ∫_0^{t_grid[j]} λ_k(s) ds.
        For t <= t_local[0], returns 0 (assuming λ=0 on [0, t_local[0]]).
        For t >= t_local[-1], returns the total integral up to t_local[-1] (flat extension).
    """
    t_local = np.asarray(t_local, dtype=float)
    lam_local = np.asarray(lam_local, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    if t_local.ndim != 1:
        raise ValueError("t_local must be 1D")
    if lam_local.ndim != 2 or lam_local.shape[0] != t_local.shape[0]:
        raise ValueError("lam_local must have shape (len(t_local), d)")
    if t_grid.ndim != 1:
        raise ValueError("t_grid must be 1D")
    if not np.all(np.diff(t_local) > 0):
        raise ValueError("t_local must be strictly increasing")
    if not np.all(np.diff(t_grid) >= 0):
        raise ValueError("t_grid must be sorted nondecreasing")

    m = t_local.shape[0]
    d = lam_local.shape[1]
    n = t_grid.shape[0]

    # Precompute slopes on each interval [t_i, t_{i+1}]
    dt = np.diff(t_local)                       # (m-1,)
    slopes = np.diff(lam_local, axis=0) / dt[:, None]   # (m-1, d)

    # Cumulative integral at knots: A[i] = ∫_0^{t_i} λ(s) ds
    # Exact for PWL via trapezoid rule
    trap = 0.5 * (lam_local[:-1] + lam_local[1:]) * dt[:, None]  # (m-1, d)
    A = np.zeros((m, d), dtype=float)
    A[1:] = np.cumsum(trap, axis=0)

    # For each t_grid[j], find i such that t_local[i] <= t < t_local[i+1]
    # np.searchsorted returns insertion index; subtract 1 to get left interval index.
    idx = np.searchsorted(t_local, t_grid, side="right") - 1  # (n,)

    L_grid = np.zeros((n, d), dtype=float)

    # Region 1: t <= t_local[0] -> integral is 0 (assuming λ=0 on [0, t_local[0]])
    left_mask = idx < 0
    # already zeros

    # Region 2: t >= t_local[-1] -> clamp to total integral up to last knot
    right_mask = idx >= (m - 1)
    if np.any(right_mask):
        L_grid[right_mask] = A[-1]

    # Region 3: interior points
    mid_mask = (~left_mask) & (~right_mask)
    if np.any(mid_mask):
        j = np.where(mid_mask)[0]
        i = idx[j]                         # interval indices, 0..m-2
        u = (t_grid[j] - t_local[i])       # (len(j),)

        # L(t) = A[i] + lam[i]*u + 0.5*slope[i]*u^2
        L_grid[j] = (
            A[i]
            + lam_local[i] * u[:, None]
            + 0.5 * slopes[i] * (u[:, None] ** 2)
        )

    return L_grid