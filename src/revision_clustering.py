"""Reproducible objective-selected restarts for vector and functional clustering.

Labels for biological evaluation are deliberately absent from this module.
"""
from time import perf_counter
import numpy as np
from sklearn.cluster import KMeans
from src.functional_l1 import landscape_distance, median_landscape


def assign_nearest(distances, previous=None):
    """Lowest-index tie initially; subsequently retain an equally good cluster."""
    assigned = np.argmin(distances, axis=1)
    if previous is not None:
        rows = np.arange(len(distances))
        ties = distances[rows, previous] == distances[rows, assigned]
        assigned[ties] = previous[ties]
    return assigned


def functional_kmedians(landscapes, clusters, seed, max_iter=300):
    if not 1 <= clusters <= len(landscapes) or max_iter < 1:
        raise ValueError("Require 1 <= clusters <= samples and max_iter >= 1")
    indices = np.random.RandomState(seed).choice(len(landscapes), clusters, replace=False)
    centres = [landscapes[i] for i in indices]

    def distances_to(current):
        return np.array([[landscape_distance(lan, centre) for centre in current]
                         for lan in landscapes])

    rows = np.arange(len(landscapes))
    distances = distances_to(centres)
    labels = assign_nearest(distances)
    history = [float(distances[rows, labels].sum())]
    converged = False
    for iteration in range(1, max_iter + 1):
        updated = []
        for j in range(clusters):
            members = [landscapes[i] for i in np.flatnonzero(labels == j)]
            # Retaining an empty centre preserves the objective descent argument.
            updated.append(median_landscape(members) if members else centres[j])
        distances = distances_to(updated)
        next_labels = assign_nearest(distances, labels)
        objective = float(distances[rows, next_labels].sum())
        if not np.isfinite(objective):
            raise ValueError("Nonfinite L1 objective")
        if objective > history[-1] + 1e-10 * max(1., history[-1]):
            raise ArithmeticError("L1 assignment/update increased the objective")
        history.append(objective)
        converged = np.array_equal(next_labels, labels)
        centres, labels = updated, next_labels
        if converged:
            break
    return {"labels": labels, "centres": centres, "objective": history[-1],
            "history": history, "iterations": iteration, "converged": converged,
            "initial_indices": indices, "occupied_clusters": len(np.unique(labels))}


def run_restarts(data, method, clusters, start_seeds, max_iter=300, euclidean_tol=1e-4):
    """Run an explicit runs-by-starts seed matrix; retain each row's minimum.

    Timings include estimator setup, fitting and selection, but exclude metrics
    and serialization. Every start's labels/objective and convergence metadata
    are retained. Exhausted iterations are flagged, never hidden as convergence.
    """
    seeds = np.asarray(start_seeds)
    if seeds.ndim != 2 or not seeds.size or method not in ("euclidean", "functional_l1"):
        raise ValueError("Expected nonempty seed matrix and supported method")
    output = []
    for run_seeds in seeds:
        start_time = perf_counter()
        starts = []
        best = None
        best_index = None
        for j, seed in enumerate(run_seeds):
            if method == "euclidean":
                model = KMeans(n_clusters=clusters, n_init=1, init="k-means++",
                               algorithm="lloyd", max_iter=max_iter, tol=euclidean_tol,
                               random_state=int(seed))
                labels = model.fit_predict(data)
                result = {"labels": labels, "centres": model.cluster_centers_,
                          "objective": float(model.inertia_), "iterations": model.n_iter_,
                          "hit_iteration_limit": model.n_iter_ >= max_iter,
                          "occupied_clusters": len(np.unique(labels))}
            else:
                result = functional_kmedians(data, clusters, int(seed), max_iter)
                result["hit_iteration_limit"] = not result["converged"]
            if not np.isfinite(result["objective"]):
                raise ValueError("Nonfinite clustering objective")
            starts.append({key: value for key, value in result.items() if key != "centres"})
            starts[-1]["seed"] = int(seed)
            if best is None or result["objective"] < best["objective"]:
                best, best_index = result, j
        output.append({"best_start": best_index, "objective": best["objective"],
                       "labels": best["labels"], "centres": best["centres"],
                       "starts": starts, "seconds": perf_counter() - start_time})
    return output
