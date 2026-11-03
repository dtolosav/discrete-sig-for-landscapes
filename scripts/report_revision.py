"""Verify saved runs and generate descriptive tables/figures (no refitting)."""
import argparse
from pathlib import Path
import sys
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.revision_reporting import write_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run-dir", type=Path, help="Summarise a freshly computed campaign")
    inputs.add_argument("--paper-bundle", type=Path, help="Regenerate all paper tables and figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / 'results/paper')
    args = parser.parse_args()
    if args.paper_bundle:
        from src.paper_reporting import report_paper
        report_paper(args.paper_bundle, args.output_dir)
    else:
        rows = write_report(args.run_dir)
        print(f"Verified and reported {len(rows)} configurations")
