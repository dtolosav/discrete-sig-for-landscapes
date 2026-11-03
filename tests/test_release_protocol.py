"""Guard the reported scope and the shared seed prefixes during release cleanup."""
import json
from pathlib import Path
import numpy as np
from src.revision_experiment import specifications, seeds_for_specification
from src.revision_features import seed_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_paper_scopes_and_repetitions():
    for cohort, count in [('x1', 6), ('x6', 23)]:
        config = json.loads((ROOT / f'configs/{cohort}.json').read_text())
        specs = specifications(config)
        assert len(specs) == count
        assert 'levels-10-functional_l1' not in [s['id'] for s in specs]
        _, seeds = seed_manifest(42, 100, 10)
        for spec in specs:
            actual = seeds_for_specification(spec, config, seeds)
            expected = 10 if cohort == 'x1' and spec['method'] == 'functional_l1' else 100
            assert actual.shape == (expected, 10)
            np.testing.assert_array_equal(actual, seeds[:expected])
            np.testing.assert_array_equal(seeds_for_specification(spec, config, seeds[:2]), seeds[:2])


def test_short_seed_matrix_preserves_published_initialisations():
    full_runs, full = seed_manifest(42, 100, 10)
    short_runs, short = seed_manifest(42, 10, 10)
    np.testing.assert_array_equal(short_runs, full_runs[:10])
    np.testing.assert_array_equal(short, full[:10])
