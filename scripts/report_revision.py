"""Verify saved runs and generate descriptive tables/figures (no refitting)."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.revision_reporting import write_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = write_report(args.run_dir)
    print(f"Verified and reported {len(rows)} configurations")
