"""Primary x1 functional L1: ten ten-start runs (D-2026-025)."""
import argparse
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import platform
import shutil
import sys
from time import perf_counter
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.input_audit import load_canonical_classifications, sha256_file
from src.process_lan import load_lan
from src.revision_experiment import bind_manifest, save_json, save_npz, run_specification
from src.revision_reporting import collect_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    session_start = perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    out = args.output_dir
    reuse = None
    if (out / 'manifest.json').exists():
        previous = json.loads((out / 'manifest.json').read_text())['identity']
        if len(previous['start_seeds']) == 100:
            reuse = out
            out = out.with_name(out.name + '-reduced-10')
            print(f'Reduced schedule: preserving {reuse}; continuing in {out}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    vector_dir = ROOT.parent / 'results/x1-comparison/2026-11-02'
    reference = json.loads((vector_dir / 'manifest.json').read_text())['identity']
    config = reference['config']
    classification = ROOT / 'data/external/trefoil_list.csv'
    assert sha256_file(classification) == reference['classification_sha256']
    _, labels, _ = load_canonical_classifications(classification)
    chains = sorted(c for c, row in labels.items()
                    if row['Representative'] and row['Representative'].lower() != 'other')
    classes = [labels[c]['Representative'] for c in chains]
    assert len(chains) == 363 and chains == reference['chains'] and classes == reference['classes']
    source = ROOT / 'data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_x1'
    hashes = {c: sha256_file(source / f'{c}_1.lan') for c in chains}
    assert hashes == reference['input_sha256']
    for relative, expected in reference['code_sha256'].items():
        if relative.startswith('src/'):
            assert sha256_file(ROOT / relative) == expected
    start = perf_counter()
    landscapes = [load_lan(source / f'{c}_1.lan', config['levels']) for c in chains]
    loading_seconds = perf_counter() - start
    assert max(x for lan in landscapes for level in lan for x, _ in level) == config['support_L']
    seeds = np.array(reference['start_seeds'])
    assert seeds.shape == (100, 10)
    seeds = seeds[:10]
    config = dict(config, runs=10)
    spec = dict(id='primary-functional_l1', group='primary', method='functional_l1',
                levels=15, weight=3, points=1024)
    files = [Path(__file__), *sorted((ROOT / 'src').glob('*.py'))]
    identity = dict(scope='full', purpose='primary x1 functional L1 comparison, ten ten-start runs, D-2026-025',
                    config=config, specifications=[spec], chains=chains, classes=classes,
                    input_sha256=hashes, classification_sha256=sha256_file(classification),
                    vector_manifest_sha256=sha256_file(vector_dir / 'manifest.json'),
                    start_seeds=seeds, python=sys.version, machine=platform.platform(), host=platform.node(),
                    versions={p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'numba')},
                    thread_environment={k: os.environ[k] for k in
                       ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')},
                    threadpools=sorted([{k: v for k, v in pool.items() if k != 'filepath'}
                                        for pool in threadpool_info()],
                                       key=lambda p: (p['internal_api'], p['prefix'])),
                    code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in files},
                    timing='One thread. Loading and functional conversion separately recorded; clustering includes all 100 starts and any first-use compilation. No signature encoding. IO, scoring and report generation excluded from numerical stages. Session wall time separately recorded; resumed setup is not counted twice.')
    if reuse is not None:
        old = json.loads((reuse / 'manifest.json').read_text())
        for key in ('chains', 'classes', 'input_sha256', 'classification_sha256', 'specifications'):
            assert old['identity'][key] == identity[key], key
        assert old['identity']['start_seeds'][:10] == seeds.tolist()
        for relative, expected in old['identity']['code_sha256'].items():
            if relative.startswith('src/'):
                assert sha256_file(ROOT / relative) == expected
        assert {k: v for k, v in old['identity']['config'].items() if k != 'runs'} == {
            k: v for k, v in config.items() if k != 'runs'}
        loading_seconds = old['measurements']['input_loading_seconds']
        identity['reused_manifest_sha256'] = sha256_file(reuse / 'manifest.json')
        identity['reused_directory'] = str(reuse)
    manifest = bind_manifest(out / 'manifest.json', identity, dict(input_loading_seconds=loading_seconds))
    for relative, expected in identity['code_sha256'].items():
        snapshot = out / 'source' / relative
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, snapshot)
        assert sha256_file(snapshot) == expected
    save_npz(out / 'seeds.npz', start_seeds=seeds)
    if reuse is not None:
        destination = out / spec['id']
        destination.mkdir(exist_ok=True)
        names = ['representation.json']
        for i, row in enumerate(seeds):
            checkpoint = reuse / spec['id'] / f'run-{i:03d}.json'
            if checkpoint.exists():
                fit = json.loads(checkpoint.read_text())
                assert [s['seed'] for s in fit['starts']] == row.tolist()
                names.extend([f'centres-{i:03d}.npz', checkpoint.name])
        for name in names:
            original, target = reuse / spec['id'] / name, destination / name
            if not target.exists():
                shutil.copyfile(original, target)
            assert sha256_file(original) == sha256_file(target)
    print('Validated primary x1 L1: 363 proteins, 15 levels, 10 runs x 10 starts; one thread.', flush=True)
    if args.prepare_only:
        return
    summary = run_specification(spec, landscapes, None, config, classes, seeds,
                                out / spec['id'], manifest['measurements']['input_loading_seconds'])
    _, rows = collect_summaries(out)
    assert summary['iteration_limit_starts'] == 0 and summary['occupied_clusters_min'] == 9
    save_json(out / 'verified_results.json', dict(rows=rows, complete=True,
              scope='primary x1 functional L1 only; combine with the separately verified vector results'))
    save_json(out / 'session_timing.json', dict(started_utc=timestamp,
              finished_utc=datetime.now(timezone.utc).isoformat(),
              wall_seconds=perf_counter() - session_start,
              note='This invocation only; numerical stage totals retained separately across resumption.'))
    print('Completed and verified primary x1 L1; see verified_results.json.', flush=True)


if __name__ == '__main__':
    main()
