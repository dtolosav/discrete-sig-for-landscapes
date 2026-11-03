"""Package the original campaign records needed to reproduce the paper outputs.

Maintainer utility: copies records unchanged, omitting restart centres and
per-protein caches. The public reporting route needs only the resulting bundle.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.input_audit import sha256_file
from src.revision_features import standardise


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('x1', 'x1-l1', 'timing', 'x6', 'x1-figures', 'x6-diagnostics', 'x6-grid-audit', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    provenance = {}

    def copy(source, relative):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        provenance[relative] = sha256_file(source)

    scopes = {}
    for name, directory in [('x1', args.x1), ('x1_l1', args.x1_l1),
                             ('timing', args.timing), ('x6', args.x6)]:
        copy(directory / 'manifest.json', name + '/manifest.json')
        identity = read(directory / 'manifest.json')['identity']
        specs = [s for s in identity['specifications'] if s['id'] != 'levels-10-functional_l1']
        if name in ('x1', 'timing'):
            specs = [s for s in specs if s['group'] == 'primary' and s['method'] != 'functional_l1']
        elif name == 'x1_l1':
            specs = [s for s in specs if s['id'] == 'primary-functional_l1']
        scopes[name] = [s['id'] for s in specs]
        for spec in specs:
            folder = directory / spec['id']
            summary = read(folder / 'summary.json')
            copy(folder / 'summary.json', f'{name}/{spec["id"]}/summary.json')
            for item in ['representation.json', *[f'run-{r:03d}.json' for r in range(summary['runs'])]]:
                source = folder / item
                assert sha256_file(source) == summary['artifact_sha256'][item], source
                copy(source, f'{name}/{spec["id"]}/{item}')
        # Keep original numerical source snapshots with the campaign provenance.
        for file in sorted((directory / 'source').rglob('*.py')):
            relative = file.relative_to(directory).as_posix()
            copy(file, f'{name}/{relative}')
    for name, directory in [('x1', args.x1), ('x6', args.x6)]:
        feature = directory / 'primary-dlfm/features.npz'
        expected = read(directory / 'primary-dlfm/summary.json')['artifact_sha256']['features.npz']
        assert sha256_file(feature) == expected
        copy(feature, f'{name}/primary-dlfm/features.npz')
        identity = read(directory / 'manifest.json')['identity']
        for spec in identity['specifications']:
            if spec['group'] != 'primary' or spec['method'] == 'functional_l1':
                continue
            folder = directory / spec['id']
            expected = read(folder / 'summary.json')['artifact_sha256']['features.npz']
            assert sha256_file(folder / 'features.npz') == expected
            with np.load(folder / 'features.npz') as saved:
                scaled, scaler, keep = standardise(saved['raw'])
                np.testing.assert_array_equal(keep, saved['keep'])
                np.testing.assert_allclose(scaled, saved['scaled'], rtol=1e-12, atol=1e-12)
                relative = f'reference/{name}-{spec["method"]}.npz'
                (out / relative).parent.mkdir(exist_ok=True)
                np.savez_compressed(out / relative, chains=np.array(identity['chains'])[[0, -1]],
                                    raw=saved['raw'][[0, -1]], scaled=scaled[[0, -1]],
                                    keep=keep, mean=scaler.mean_, scale=scaler.scale_)
            provenance[relative] = sha256_file(out / relative)
    if (args.timing / 'timing_results.json').exists():
        copy(args.timing / 'timing_results.json', 'timing/timing_results.json')
    for name in ('pca_coordinates.npz', 'figure_metadata.json', 'uniform_grid_audit.json'):
        copy(args.x1_figures / name, 'reference/' + name)
    copy(args.x6_diagnostics / '4qef_membership.csv', 'reference/x6_membership.csv')
    copy(args.x6_grid_audit, 'reference/x6_grid_audit.json')
    copy(ROOT / 'data/external/trefoil_list.csv', 'reference/trefoil_list.csv')
    index = dict(format_version=1, scopes=scopes, sha256=provenance,
                 note='Original files copied unchanged. Restart centres and most feature caches are omitted; their historical hashes remain in the original manifests.')
    (out / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    archive = Path(str(out) + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname=out.name)
    digest = sha256_file(archive)
    archive.with_suffix(archive.suffix + '.sha256').write_text(f'{digest}  {archive.name}\n')
    print(f'{archive}: {archive.stat().st_size:,} bytes; SHA256 {digest}')


if __name__ == '__main__':
    main()
