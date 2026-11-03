"""Fetch the two published landscape cohorts at the audited source commit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/katherine-benjamin/ph-knotted-proteins.git'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=('x1', 'x6', 'both'), default='both')
    args = parser.parse_args()
    checks = json.loads((ROOT / 'data/external/input_checksums.json').read_text())
    destination = ROOT / 'data/external/landscapes/ph-knotted-proteins'
    cohorts = ('x1', 'x6') if args.cohort == 'both' else (args.cohort,)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', URL, str(destination)], check=True)
        # A sparse checkout avoids downloading point clouds and diagrams.
        subprocess.run(['git', '-C', str(destination), 'sparse-checkout', 'set',
                        'data/validation/landscapes/interpolated_x1',
                        'data/validation/landscapes/interpolated_x6'], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', checks['source_commit']], check=True)
    commit = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == checks['source_commit'], 'Existing data checkout has a different commit'
    for cohort in cohorts:
        directory = destination / f'data/validation/landscapes/interpolated_{cohort}'
        expected = checks['cohorts'][cohort]
        assert {p.name for p in directory.glob('*_1.lan')} == set(expected)
        for filename, digest in expected.items():
            assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == digest, filename
        print(f'{cohort}: verified all {len(expected)} degree-one landscapes', flush=True)


if __name__ == '__main__':
    main()
