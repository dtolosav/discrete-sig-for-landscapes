import numpy as np
import pytest
from src.functional_l1 import landscape_distance, median_curve, sample
from src.revision_clustering import assign_nearest, functional_kmedians, run_restarts


def triangles(heights):
    return [[(np.array([0., 1., 2.]), np.array([0., h, 0.]))] for h in heights]


def test_exact_median_clustering_and_objective():
    data = triangles([0., 0., 1., 9., 10., 10.])
    fit = functional_kmedians(data, 2, 42)
    assert fit["converged"]
    assert fit["objective"] == pytest.approx(2.)
    assert np.all(np.diff(fit["history"]) <= 1e-12)
    assert len(set(fit["labels"][:3])) == 1
    assert len(set(fit["labels"][3:])) == 1
    assert fit["labels"][0] != fit["labels"][-1]
    objective = sum(landscape_distance(x, fit["centres"][j])
                    for x, j in zip(data, fit["labels"]))
    assert objective == pytest.approx(fit["objective"])


def test_duplicate_inputs_empty_centres_and_ties():
    fit = functional_kmedians(triangles([1., 1., 1.]), 3, 1)
    assert fit["converged"] and fit["objective"] == 0
    assert fit["occupied_clusters"] == 1
    np.testing.assert_array_equal(assign_nearest(np.ones((2, 3))), [0, 0])
    np.testing.assert_array_equal(assign_nearest(np.ones((2, 3)), np.array([2, 1])), [2, 1])


@pytest.mark.parametrize("method", ["euclidean", "functional_l1"])
def test_restarts_are_reproducible_and_selected_by_objective(method):
    data = np.array([[0.], [1.], [2.], [7.], [9.], [10.]])
    if method == "functional_l1":
        data = triangles(data[:, 0])
    seeds = [[1, 2, 3], [4, 5, 6]]
    first = run_restarts(data, method, 2, seeds)
    second = run_restarts(data, method, 2, seeds)
    for a, b in zip(first, second):
        assert a["objective"] == min(s["objective"] for s in a["starts"])
        assert a["best_start"] == b["best_start"]
        np.testing.assert_array_equal(a["labels"], b["labels"])
        for s, t in zip(a["starts"], b["starts"]):
            np.testing.assert_array_equal(s["labels"], t["labels"])


def test_iteration_cap_is_reported():
    fit = functional_kmedians(triangles([0., 1., 2., 3., 20., 21.]), 2, 0, max_iter=1)
    assert fit["iterations"] == 1
    assert "converged" in fit
    with pytest.raises(ValueError):
        functional_kmedians(triangles([0.]), 2, 0)


def test_vectorised_median_matches_dense_pointwise_median():
    rng = np.random.RandomState(17)
    knots = np.array([0., .2, .7, 1.])
    curves = [(knots, np.r_[0., rng.rand(2), 0.]) for _ in range(7)]
    median = median_curve(curves)
    query = np.linspace(0, 1, 1001)
    np.testing.assert_allclose(sample(median, query),
                               np.median([sample(c, query) for c in curves], axis=0),
                               atol=1e-14)
