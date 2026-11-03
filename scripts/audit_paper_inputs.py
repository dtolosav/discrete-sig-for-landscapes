"""Recompute landscape approximation errors and the x6 within-class comparison.

These are descriptive checks reported in the paper, not new clustering runs.
The saved bundle supplies the original partitions; landscapes come from the
pinned upstream input checkout.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.process_lan import load_lan
from src.revision_features import evaluate, reconstruction_errors
from src.functional_l1 import as_functional, l1_distance
from src.revision_experiment import save_json


def landscape_audit(cohort, identity, source):
    landscapes = [load_lan(source / f'{chain}_1.lan', 15) for chain in identity['chains']]
    support = identity['config']['support_L']
    grid = np.linspace(0, support, 1024)
    values = evaluate(landscapes, grid)
    errors = reconstruction_errors(landscapes, grid)
    functions = as_functional(landscapes, support)
    areas = np.array([[float(np.trapz(y, x)) for x, y in levels] for levels in functions])
    absolute = np.array([sum(l1_distance(curve, (grid, values[i, j])) for j, curve in enumerate(levels))
                         for i, levels in enumerate(functions)])
    return dict(cohort=cohort, chains=identity['chains'], support_L=support,
                points=1024, max_sup_error=float(errors.max()),
                max_relative_L1_error=float((absolute / areas.sum(1)).max()),
                sup_error=errors, L1_error=absolute, areas=areas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = json.loads((ROOT / 'data/external/input_checksums.json').read_text())
    from src.input_audit import sha256_file
    for cohort in ('x1', 'x6'):
        identity = json.loads((args.paper_bundle / cohort / 'manifest.json').read_text())['identity']
        source = ROOT / f'data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_{cohort}'
        for chain in identity['chains']:
            path = source / f'{chain}_1.lan'
            assert sha256_file(path) == checks['cohorts'][cohort][path.name]
        report = landscape_audit(cohort, identity, source)
        save_json(args.output_dir / f'{cohort}_landscape_audit.json', report)
        print(cohort, report['max_sup_error'], report['max_relative_L1_error'], flush=True)
    # The CSV records original per-chain measurements, not just group medians.
    import csv
    with (args.paper_bundle / 'reference/x6_membership.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    indices = {chain: i for i, chain in enumerate(report['chains'])}
    for row in rows:
        np.testing.assert_allclose(report['areas'][indices[row['Chain']]][10],
                                   float(row['Level 11 area']), rtol=1e-12, atol=1e-12)
    print('Verified all 223 original level-11 areas against the pinned landscapes.')


if __name__ == '__main__':
    main()
