"""Verify the author-approved 23-configuration x6 scope (D-2026-021).

The original 24-configuration manifest is preserved. Only the unfinished
ten-level functional L1 ablation is excluded, explicitly, from this report.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.input_audit import sha256_file
from src.revision_reporting import write_report
from src.revision_experiment import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((campaign / 'manifest.json').read_text())
    original = manifest['identity']['specifications']
    deferred = 'levels-10-functional_l1'
    assert len(original) == 24 and sum(s['id'] == deferred for s in original) == 1
    selected = [s for s in original if s['id'] != deferred]
    manifest['identity']['specifications'] = selected
    with tempfile.TemporaryDirectory(prefix='discrete-sig-reduced-x6-') as temp:
        view = Path(temp)
        save_json(view / 'manifest.json', manifest)
        for spec in selected:
            (view / spec['id']).symlink_to(campaign / spec['id'], target_is_directory=True)
        rows = write_report(view)
        for row in rows:
            assert row['runs'] == 100 and row['initialisations'] == 10
            assert row['iteration_limit_starts'] == 0, row['id']
            assert row['occupied_clusters_min'] == 9, row['id']
        shutil.copytree(view / 'reports', output / 'reports', dirs_exist_ok=True)
    save_json(output / 'scope.json', dict(
        decision='D-2026-021', original_configurations=24, reported_configurations=23,
        completed=[s['id'] for s in selected], deferred=[deferred],
        reason='Author-approved runtime reduction on 2026-11-02; partial L1 scores excluded.',
        campaign_manifest_sha256=sha256_file(campaign / 'manifest.json'),
        reporting_code_sha256=sha256_file(ROOT / 'src/revision_reporting.py'),
        selection_script_sha256=sha256_file(Path(__file__)), complete=True))
    print('Verified 23 configurations. Ten-level L1 remains deferred; original manifest unchanged.', flush=True)


if __name__ == '__main__':
    main()
