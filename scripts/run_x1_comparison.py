"""Author-approved x1 comparison: five primary vector methods, no ablations.

Reuse audited diagnostic DLFM features; run all 100 ten-start clusterings with
the existing revision runner. Outputs are separate from the x6 campaign.
"""
import argparse
import os
from pathlib import Path
import json
import shutil
import sys
from importlib.metadata import version
from time import perf_counter

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.input_audit import load_canonical_classifications, sha256_file
from src.process_lan import load_lan
from src.revision_features import critical_grid, seed_manifest, standardise, evaluate, discrete_features
from src.revision_experiment import bind_manifest, save_json, save_npz, specifications, run_specification
from src.revision_reporting import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT.parent / 'results/x1-diagnostic/2026-11-02'
    old = json.loads((cache / 'manifest.json').read_text())['identity']
    config = json.loads((ROOT / 'configs/revision.json').read_text())
    config.update(cohort='x1', support_L=old['support'], ablation_levels=[],
                  ablation_weights=[], ablation_grid_points=[])
    specs = [s for s in specifications(config) if s['method'] != 'functional_l1']
    source = ROOT / 'data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_x1'
    classification = ROOT / 'data/external/trefoil_list.csv'
    _, labels, _ = load_canonical_classifications(classification)
    chains = sorted(c for c, row in labels.items()
                    if row['Representative'] and row['Representative'].lower() != 'other')
    classes = [labels[c]['Representative'] for c in chains]
    assert len(chains) == 363 and chains == old['chains'] and classes == old['classes']
    assert sha256_file(classification) == old['classification_sha256']
    hashes = {c: sha256_file(source / f'{c}_1.lan') for c in chains}
    assert hashes == old['input_sha256']
    for relative, expected in old['code_sha256'].items():
        if relative.startswith('discrete-sig-for-landscapes/src/'):
            assert sha256_file(ROOT.parent / relative) == expected
    start = perf_counter()
    landscapes = [load_lan(source / f'{c}_1.lan', 15) for c in chains]
    load_seconds = perf_counter() - start
    support = max(x for lan in landscapes for level in lan for x, _ in level)
    assert support == config['support_L']
    start = perf_counter()
    grid = critical_grid(landscapes, support)
    grid_seconds = perf_counter() - start
    import hashlib
    assert hashlib.sha256(grid.tobytes()).hexdigest() == old['grid_sha256']
    run_seeds, seeds = seed_manifest(42, 100, 10)
    assert np.array_equal(seeds[:50], old['start_seeds'])
    x6 = json.loads((ROOT.parent / 'results/revision-full/2026-10-28-run/manifest.json').read_text())['identity']
    assert chains == x6['chains'] and classes == x6['classes']
    assert np.array_equal(seeds, x6['start_seeds'])
    with np.load(cache / 'features.npz') as stored:
        raw = stored['corrected'].copy()
        original = stored['raw']
        first = stored['first_order_columns']
        expected = original.copy()
        expected[:, first] = 0
        assert np.array_equal(raw, expected) and raw.shape == (363, 8015)
    assert np.isfinite(raw).all()
    code = [Path(__file__), *sorted((ROOT / 'src').glob('*.py'))]
    identity = dict(scope='full', purpose='author-approved x1 primary vector sensitivity comparison',
                    config=config, specifications=specs, chains=chains, classes=classes,
                    depth=[float(labels[c]['Depth']) for c in chains],
                    input_sha256=hashes, classification_sha256=sha256_file(classification),
                    source_commit=config['source_commit'], critical_grid=grid,
                    run_seeds=run_seeds, start_seeds=seeds,
                    versions={p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'numba', 'pysiglib', 'torch')},
                    python=sys.version, code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in code},
                    diagnostic_manifest_sha256=sha256_file(cache / 'manifest.json'),
                    cached_features_sha256=sha256_file(cache / 'features.npz'),
                    timing_note='Sensitivity scores only: concurrent x6 work; imported DLFM row times combine evaluation and encoding, exclude original setup. Not manuscript runtime benchmarks.')
    manifest = bind_manifest(out / 'manifest.json', identity,
                             dict(input_loading_seconds=load_seconds, critical_grid_seconds=grid_seconds))
    for relative, expected_hash in identity['code_sha256'].items():
        snapshot = out / 'source' / relative
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, snapshot)
        assert sha256_file(snapshot) == expected_hash
    save_npz(out / 'grid_and_seeds.npz', grid=grid, run_seeds=run_seeds, start_seeds=seeds)
    # Recompute one full-grid row to check compatibility before importing caches.
    if not (out / 'cache_validation.json').exists():
        check = discrete_features(evaluate(landscapes[:1], grid), 3)[0]
        np.testing.assert_allclose(check, raw[0], rtol=1e-12, atol=1e-12)
        save_json(out / 'cache_validation.json', dict(chain=chains[0], passed=True,
                  max_absolute_difference=float(np.max(np.abs(check - raw[0])))))
    folder = out / 'primary-dlfm'
    folder.mkdir(exist_ok=True)
    if not (folder / 'representation.json').exists():
        start = perf_counter()
        scaled, scaler, keep = standardise(raw)
        scaling_seconds = perf_counter() - start
        assert np.isfinite(scaled).all() and scaled.shape[1] == 8000
        save_npz(folder / 'features.npz', raw=raw, scaled=scaled, keep=keep,
                 mean=scaler.mean_, scale=scaler.scale_)
        original_seconds = 0.
        for i in range(len(chains)):
            with np.load(cache / 'rows' / f'{i:04d}.npz') as row:
                np.testing.assert_array_equal(row['raw'], original[i])
                original_seconds += float(row['seconds'])
        save_json(folder / 'representation.json', dict(preprocessing_seconds=0.,
                  representation_seconds=original_seconds + scaling_seconds, warmup_seconds=0.,
                  raw_dimension=raw.shape[1], retained_dimension=scaled.shape[1],
                  features_sha256=sha256_file(folder / 'features.npz')))
    print(f'Validated x1: {len(chains)} chains, {len(grid)} grid points, L={support}; five methods, 100 x 10 starts.', flush=True)
    if args.prepare_only:
        return
    for spec in specs:
        shared = manifest['measurements']['input_loading_seconds']
        if spec['method'] != 'flattened':
            shared += manifest['measurements']['critical_grid_seconds']
        run_specification(spec, landscapes, grid, config, classes, seeds, out / spec['id'], shared)
        completed = [s['id'] for s in specs if (out / s['id'] / 'summary.json').exists()]
        save_json(out / 'status.json', dict(completed=completed, expected=5, complete=len(completed) == 5))
    write_report(out)
    print('Completed and verified all five methods; see reports/comparisons.csv.', flush=True)


if __name__ == '__main__':
    main()
