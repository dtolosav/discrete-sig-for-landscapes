"""Serial x1 timing: fresh full-cohort features, ten ten-start runs per method."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
from importlib.metadata import version

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.input_audit import sha256_file
from src.process_lan import load_lan
from src.revision_features import critical_grid
from src.revision_experiment import bind_manifest, save_json, run_specification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    reference = ROOT.parent / 'results/x1-comparison/2026-11-02'
    identity = json.loads((reference / 'manifest.json').read_text())['identity']
    source = ROOT / 'data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_x1'
    for chain, expected in identity['input_sha256'].items():
        assert sha256_file(source / f'{chain}_1.lan') == expected
    for name, expected in identity['code_sha256'].items():
        if name.startswith('src/'):
            assert sha256_file(ROOT / name) == expected
    assert sha256_file(ROOT / 'data/external/trefoil_list.csv') == identity['classification_sha256']
    seeds = np.array(identity['start_seeds'][:10])
    assert seeds.shape == (10, 10) and len(identity['chains']) == 363
    specs = identity['specifications']
    assert [s['method'] for s in specs] == ['dlfm', 'chen', 'time_chen', 'integrated_chen', 'flattened']
    print('Validated inputs, unchanged numerical sources, five methods, 363 chains, 10 x 10 seeds.', flush=True)
    if args.prepare_only:
        return
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = identity['config']
    start = perf_counter()
    landscapes = [load_lan(source / f'{c}_1.lan', 15) for c in identity['chains']]
    loading = perf_counter() - start
    start = perf_counter()
    grid = critical_grid(landscapes, config['support_L'])
    grid_seconds = perf_counter() - start
    np.testing.assert_array_equal(grid, identity['critical_grid'])
    files = [Path(__file__), *sorted((ROOT / 'src').glob('*.py'))]
    manifest = bind_manifest(out / 'manifest.json', dict(
        purpose='Author-approved serial x1 timing; not replacement clustering scores',
        reference_manifest_sha256=sha256_file(reference / 'manifest.json'),
        config=config, specifications=specs, start_seeds=seeds,
        chains=identity['chains'], input_sha256=identity['input_sha256'],
        python=sys.version, machine=platform.platform(), host=platform.node(),
        versions={p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'numba', 'pysiglib', 'torch')},
        threads=1, encoding_repetitions=1, timing_runs=10, starts_per_run=10,
        code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in files},
        boundaries='Supplied landscapes onward; one fresh full-cohort encoding per method. Loading/grid charged to each applicable method. Tiny encoding warmup separate. Clustering includes estimator setup and all ten starts per run. Excludes input hashing, IO, scoring, validation and plotting. No estimated 100-run totals.'),
        dict(input_loading_seconds=loading, critical_grid_seconds=grid_seconds))
    for p in files:
        target = out / 'source' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(p, target)
        assert sha256_file(target) == sha256_file(p)
    rows = []
    for spec in specs:
        shared = manifest['measurements']['input_loading_seconds']
        if spec['method'] != 'flattened':
            shared += manifest['measurements']['critical_grid_seconds']
        folder = out / spec['id']
        summary = run_specification(spec, landscapes, grid, config, identity['classes'], seeds, folder, shared)
        # Validation is outside all measured stages. Existing scores remain authoritative.
        with np.load(folder / 'features.npz') as actual, np.load(reference / spec['id'] / 'features.npz') as expected:
            for key in ('raw', 'scaled'):
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        times = []
        for i in range(10):
            fit = json.loads((folder / f'run-{i:03d}.json').read_text())
            prior = json.loads((reference / spec['id'] / f'run-{i:03d}.json').read_text())
            assert [s['seed'] for s in fit['starts']] == seeds[i].tolist()
            assert fit['labels'] == prior['labels']
            times.append(fit['seconds'])
        assert summary['iteration_limit_starts'] == 0 and summary['occupied_clusters_min'] == 9
        rows.append(dict(method=spec['method'], preprocessing_seconds=shared + summary['preprocessing_seconds'],
                         representation_seconds=summary['representation_seconds'], warmup_seconds=summary['warmup_seconds'],
                         clustering_run_seconds=times, clustering_median_seconds=float(np.median(times)),
                         clustering_q25_seconds=float(np.quantile(times, .25)),
                         clustering_q75_seconds=float(np.quantile(times, .75)),
                         raw_dimension=summary['raw_dimension'], retained_dimension=summary['retained_dimension']))
        save_json(out / 'timing_results.json', dict(complete=len(rows) == 5, rows=rows,
                  note='One encoding measurement per method; clustering summaries use ten ten-start runs.'))
        print(f"Verified timing: {spec['id']}", flush=True)
    print('All five fresh encodings and timing comparisons verified.', flush=True)


if __name__ == '__main__':
    main()
