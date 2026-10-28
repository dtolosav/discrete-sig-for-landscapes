import numpy as np
import pytest
from src.revision_features import (critical_grid, evaluate, reconstruction_errors,
                                   chen_features, discrete_features, standardise,
                                   seed_manifest)
from src.functional_l1 import l1_distance, median_curve, median_landscape


def test_sampling_and_reconstruction():
    landscapes = [[[(0., 0.), (1., 1.), (2., 0.)], []]]
    grid = critical_grid(landscapes, 2)
    np.testing.assert_array_equal(evaluate(landscapes, grid), [[[0, 1, 0], [0, 0, 0]]])
    assert reconstruction_errors(landscapes, grid)[0] == 0
    assert reconstruction_errors(landscapes, np.array([0., 2.]))[0] == 1
    with pytest.raises(ValueError):
        critical_grid(landscapes, 1)


def test_chen_known_path_and_refinement():
    values = np.array([[[0., 1., 1.], [0., 0., 1.]]])
    np.testing.assert_allclose(chen_features(values, [0, 1, 2], 2)[0],
                               [1, 1, .5, 1, 0, .5])
    coarse = np.array([[[0., 1., 0.]]])
    fine = np.array([[[0., .5, 1., .5, 0.]]])
    np.testing.assert_allclose(chen_features(coarse, [0, 1, 2], 3, True),
                               chen_features(fine, [0, .5, 1, 1.5, 2], 3, True), atol=1e-14)
    assert chen_features(coarse, [0, 1, 2], 3, True).shape == (1, 14)


def test_discrete_refinement_and_zero_padding():
    a = discrete_features(np.array([[[0., 1., 0.]]]), 2)
    b = discrete_features(np.array([[[0., .5, 1., .5, 0.]]]), 2)
    c = discrete_features(np.array([[[0., 0., 1., 0., 0.]]]), 2)
    np.testing.assert_allclose(a, c)
    assert not np.allclose(a, b)
    assert a.shape == (1, 3)
    # One-letter Z^2 and two-letter Z|Z are respectively 2 and -1.
    np.testing.assert_allclose(np.sort(a[0]), [-1, 0, 2])


def test_short_bar_fixed_grid_converges_while_critical_grid_refines():
    # Barcode [0,2] plus [.5-epsilon,.5+epsilon]: the small tent is
    # below the large one, hence is exactly the second landscape level.
    triangle = [(0., 0.), (1., 1.), (2., 0.)]
    base = [[triangle, []]]
    fixed = np.array([0., .25, .5, .75, 1., 2.])
    base_fixed = discrete_features(evaluate(base, fixed), 2)
    base_critical = discrete_features(evaluate(base, critical_grid(base, 2.)), 2)
    differences = []
    for epsilon in (.01, .001):
        perturbed = [[triangle, [(.5-epsilon, 0.), (.5, epsilon), (.5+epsilon, 0.)]]]
        grid = critical_grid(perturbed, 2.)
        sampled = evaluate(perturbed, fixed)
        assert np.max(np.abs(sampled - evaluate(base, fixed))) == pytest.approx(epsilon)
        differences.append(np.linalg.norm(discrete_features(sampled, 2) - base_fixed))
        refined = discrete_features(evaluate(perturbed, grid), 2)
        # First FRUITS coordinate at dimension/weight 2 is the letter Z_1^2.
        assert abs(refined[0, 0] - base_critical[0, 0]) > .49
    assert differences[1] < .2 * differences[0]


def test_scaler_and_seeds():
    z, scaler, keep = standardise([[1, 2, 9], [3, 4, 9], [5, 6, 9]])
    np.testing.assert_array_equal(keep, [True, True, False])
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(z.var(0), 1)
    np.testing.assert_allclose(scaler.transform([[1, 2]]), z[:1])
    a, b = seed_manifest(42, 100, 10)
    c, d = seed_manifest(42, 100, 10)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)


def test_discrete_degree_one_telescopes_exactly():
    import fruits
    values = np.random.RandomState(17).normal(size=(3, 10, 31))
    values[:, :, -1] = values[:, :, 0]
    iss = fruits.ISS(fruits.words.of_weight(2, dim=10), mode=fruits.ISSMode.EXTENDED)
    columns = {iss.label(i): i for i in range(iss.n_iterated_sums())}
    indices = [columns[f"[{j}]"] for j in range(1, 10)] + [columns["[(10)]"]]
    np.testing.assert_array_equal(discrete_features(values, 2)[:, indices], np.zeros((3, 10)))
    values[:, :, -1] += .125
    np.testing.assert_array_equal(discrete_features(values, 2)[:, indices],
                                  values[:, :, -1] - values[:, :, 0])


def test_discrete_plan_reuse_does_not_retain_previous_input_values():
    from src.revision_features import _discrete_plan
    rng = np.random.RandomState(61)
    first = rng.normal(size=(2, 3, 9))
    second = rng.normal(size=(1, 3, 13))
    _discrete_plan.cache_clear()
    expected = discrete_features(second, 3)
    _discrete_plan.cache_clear()
    discrete_features(first, 3)
    actual = discrete_features(second, 3)
    np.testing.assert_array_equal(actual, expected)
    assert _discrete_plan.cache_info().hits == 1


def test_closed_path_coordinates_are_exact_constants_before_scaling():
    values = np.array([[[0., 1., 0.]], [[0., .3, 0.]], [[0., 2., 0.]]])
    plain = chen_features(values, [0, .7, 2], 3)
    np.testing.assert_array_equal(plain, np.zeros((3, 3)))
    augmented = chen_features(values, [0, .7, 2], 3, True)
    # tt, ll, ttt, lll, plus the first-level coordinates.
    constant = [0, 1, 2, 5, 6, 13]
    for column in constant:
        assert np.ptp(augmented[:, column]) == 0


def test_l1_crossing_integral():
    a = (np.array([0., 1., 2., 3.]), np.array([0., 1., 0., 0.]))
    b = (np.array([0., 1., 2., 3.]), np.array([0., 0., 1., 0.]))
    assert l1_distance(a, b) == pytest.approx(1.5)
    assert l1_distance(a, a) == 0
    assert l1_distance(a, b) == l1_distance(b, a)


def test_median_interior_crossing_objective():
    x = np.array([0., 1., 2., 3.])
    curves = [(x, np.array(y)) for y in ([0., 1., 0., 0.],
                                       [0., 0., 1., 0.], [0., 0., 0., 0.])]
    median = median_curve(curves)
    assert 1.5 in median[0]
    assert np.interp(1.5, *median) == pytest.approx(.5)
    objective = sum(l1_distance(c, median) for c in curves)
    assert objective <= min(sum(l1_distance(c, candidate) for c in curves)
                            for candidate in curves)
    # Independently integrate the pointwise absolute deviations on each interval.
    assert objective == pytest.approx(1.75)


def test_even_singleton_and_missing_levels():
    x = np.array([0., 1., 2.])
    a, b = (x, np.array([0., 1., 0.])), (x, np.zeros(3))
    np.testing.assert_allclose(median_curve([a])[1], a[1])
    np.testing.assert_allclose(median_curve([a, b])[1], [0, .5, 0])
    centre = median_landscape([[a], [b, a]])
    assert len(centre) == 2
    assert np.interp(1, *centre[1]) == .5


def test_historical_median_adapter(monkeypatch):
    import runpy
    from pathlib import Path
    from src import landscapes as pl
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    functions = runpy.run_path(str(root / "scripts/3c_kmeans_on_landscapes_L1.py"))
    knots = np.array([-np.inf, 0., 1., 2., 3., np.inf])
    inputs = [pl.PersistenceLandscape([knots], [np.array(y)]) for y in
              ([0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 0., 0.])]
    centre = functions["_median_landscape"](inputs)
    assert centre.lambda_k(1, 1.5) == pytest.approx(.5)
    assert sum(pl.distance(c, centre, 1) for c in inputs) == pytest.approx(1.75)


def test_functional_loader_keeps_final_level(tmp_path):
    from src import landscapes as pl
    path = tmp_path / "triangle.lan"
    path.write_text("#lambda_1\n0 0\n1 1\n2 0\n")
    landscape = pl.load(path)
    assert landscape.max_k() == 1
    assert landscape.lambda_k(1, 1) == 1
    assert landscape.norm(1) == pytest.approx(1)


def test_repeated_identical_knots_preserve_the_landscape(tmp_path):
    from src.process_lan import load_lan
    path = tmp_path / "repeated.lan"
    path.write_text("#lambda_1\n0 0\n1 1\n2 0\n2 0\n2 0\n3 0\n")
    levels = load_lan(path, 1)
    assert levels == [[(0., 0.), (1., 1.), (2., 0.), (3., 0.)]]
    np.testing.assert_array_equal(evaluate([levels], np.array([0., 1., 2., 3.])),
                                  [[[0., 1., 0., 0.]]])
    # A conflicting height at the same abscissa is not silently repaired.
    path.write_text("#lambda_1\n0 0\n1 1\n1 2\n2 0\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate([load_lan(path, 1)], np.array([0., 1., 2.]))
