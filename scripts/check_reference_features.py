"""Recompute selected proteins on the full paper grids and compare coordinates.

Tolerances are fixed at rtol=atol=1e-12. This is a numerical regression check,
not a substitute for recomputing every protein and every clustering run.
"""
import argparse
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.process_lan import load_lan
from src.revision_features import evaluate
from src.revision_experiment import encode
from src.input_audit import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-bundle', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.paper_bundle
    checks = json.loads((bundle / 'index.json').read_text())['sha256']
    for cohort in ('x1', 'x6'):
        identity = json.loads((bundle / cohort / 'manifest.json').read_text())['identity']
        source = ROOT / f'data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_{cohort}'
        for spec in identity['specifications']:
            if spec['group'] != 'primary' or spec['method'] == 'functional_l1':
                continue
            relative = f'reference/{cohort}-{spec["method"]}.npz'
            assert sha256_file(bundle / relative) == checks[relative]
            with np.load(bundle / relative) as expected:
                grid = (np.linspace(0, identity['config']['support_L'], spec['points'])
                        if spec['method'] == 'flattened' else np.array(identity['critical_grid']))
                for i, chain in enumerate(expected['chains']):
                    path = source / f'{chain}_1.lan'
                    assert sha256_file(path) == identity['input_sha256'][str(chain)]
                    landscape = load_lan(path, spec['levels'])
                    raw = encode(evaluate([landscape], grid), grid, spec)[0]
                    np.testing.assert_allclose(raw, expected['raw'][i], rtol=1e-12, atol=1e-12)
                    scaled = (raw[expected['keep']] - expected['mean']) / expected['scale']
                    np.testing.assert_allclose(scaled, expected['scaled'][i], rtol=1e-12, atol=1e-12)
            print(cohort, spec['method'], 'full-grid reference rows match', flush=True)


if __name__ == '__main__':
    main()
