import numpy as np
import pytest
from src.revision_reporting import depth_pairs, rank_correlation


def test_depth_pairs_are_unique_and_partition_within_between():
    features = np.array([[0., 0.], [3., 4.], [0., 4.], [3., 0.]])
    pairs, stats = depth_pairs(features, [.1, .2, .3, .4], ["a", "a", "b", "b"])
    assert len(pairs["i"]) == 6
    assert np.all(pairs["i"] < pairs["j"])
    np.testing.assert_allclose(pairs["signature_distance"], [5, 4, 3, 3, 4, 5])
    assert stats["within_classes"]["pairs"] == 2
    assert stats["between_classes"]["pairs"] == 4
    assert stats["class:a"]["spearman_rho"] is None
    assert rank_correlation([1, 2, 2, 4], [4, 3, 3, 1]) == pytest.approx(-1)
    assert rank_correlation([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(ValueError):
        depth_pairs(features, [.1], ["a"])
