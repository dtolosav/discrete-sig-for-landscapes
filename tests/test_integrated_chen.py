import numpy as np
from src.integrated_chen import integrated_chen_features
from src.integrate import integrate_lan_on_grid
from pysiglib import sig


def test_integral_triangle_in_one_dimension():
    values = np.array([[[0., 1., 0.]]])
    np.testing.assert_allclose(integrated_chen_features(values, [0, 1, 2], 3),
                               [[1., .5, 1/6]], atol=1e-14)


def test_quadratic_path_analytic_second_level():
    # Integral path is (t^2/2, t-t^2/2), not the line joining its endpoints.
    values = np.array([[[0., 1.], [1., 0.]]])
    expected = [[.5, .5, 1/8, 1/24, 5/24, 1/8]]
    np.testing.assert_allclose(integrated_chen_features(values, [0, 1], 2),
                               expected, atol=1e-14)


def test_refining_derivative_does_not_change_integrated_signature():
    grid = np.array([0., .3, .65, 1.])
    path = np.array([[0., 0.], [.3, .2], [.35, .1], [0., 0.]])
    refined = np.sort(np.r_[grid, (grid[:-1] + grid[1:]) / 2])
    fine = np.array([np.interp(refined, grid, path[:, j]) for j in range(2)])
    exact = integrated_chen_features(path.T[None], grid, 3)
    np.testing.assert_allclose(exact, integrated_chen_features(fine[None], refined, 3),
                               atol=1e-14)
    errors = []
    for n in (20, 80, 320):
        query = np.unique(np.r_[np.linspace(0, 1, n), grid])
        integrated = integrate_lan_on_grid(grid, path, query)
        approximation = np.asarray(sig(np.array(integrated, order="C", copy=True), 3))
        errors.append(np.max(np.abs(approximation - exact[0])))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-5
