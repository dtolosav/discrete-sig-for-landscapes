"""Exact PWL L1 geometry and pointwise median centres.

Each curve is (knots, values), finite with zero endpoint values and zero
extension. These centres need not be landscapes realised by barcodes.
"""
import numpy as np


def sample(curve, knots):
    return np.interp(knots, curve[0], curve[1], left=0, right=0)


def l1_distance(a, b):
    knots = np.union1d(a[0], b[0])
    diff = sample(a, knots) - sample(b, knots)
    left, right = diff[:-1], diff[1:]
    widths = np.diff(knots)
    area = widths * (np.abs(left) + np.abs(right)) / 2
    cross = left * right < 0
    area[cross] = (widths[cross] * (left[cross]**2 + right[cross]**2)
                   / (2 * (np.abs(left[cross]) + np.abs(right[cross]))))
    return float(area.sum())


def median_curve(curves):
    """Include interior crossings; even clusters use mean of middle values.

    This reference implementation checks every pair, vectorising over intervals.
    Its cost must be assessed before full-cohort clustering.
    """
    if not curves:
        raise ValueError("Cannot compute the median of an empty cluster")
    knots = np.unique(np.concatenate([c[0] for c in curves]))
    values = np.array([sample(c, knots) for c in curves])
    # With one or two members the (midpoint) median is affine on each interval.
    if len(curves) <= 2:
        return knots, np.mean(values, axis=0)
    crossings = [knots]
    widths = np.diff(knots)
    for a in range(len(curves)):
        for b in range(a):
            diff = values[a] - values[b]
            cross = diff[:-1] * diff[1:] < 0
            crossings.append(knots[:-1][cross] + widths[cross] * diff[:-1][cross]
                             / (diff[:-1][cross] - diff[1:][cross]))
    knots = np.unique(np.concatenate(crossings))
    values = np.median([sample(c, knots) for c in curves], axis=0)
    return knots, values


def median_landscape(landscapes):
    if not landscapes:
        raise ValueError("Cannot compute the median of an empty cluster")
    d = max(map(len, landscapes))
    zero = (np.array([0., 1.]), np.array([0., 0.]))
    return [median_curve([lan[r] if r < len(lan) else zero for lan in landscapes])
            for r in range(d)]


def landscape_distance(a, b):
    """Integrated L1 over all supplied levels, zero extending missing levels."""
    zero = (np.array([0., 1.]), np.zeros(2))
    return sum(l1_distance(a[r] if r < len(a) else zero,
                           b[r] if r < len(b) else zero)
               for r in range(max(len(a), len(b))))


def as_functional(landscapes, support):
    """Convert load_lan output; empty retained levels become zero curves."""
    zero = (np.array([0., support]), np.zeros(2))
    return [[tuple(np.asarray(points, dtype=float).T) if points else zero
             for points in levels] for levels in landscapes]
