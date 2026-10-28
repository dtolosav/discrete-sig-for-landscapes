#!/usr/bin/env python3
"""Create a reproducible manifest for raw persistence-landscape inputs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.input_audit import audit_inputs


def parse_cohort(value):
    try:
        name, directory = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use NAME=PATH") from exc
    if not name or not directory:
        raise argparse.ArgumentTypeError("Use NAME=PATH")
    path = Path(directory)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Cohort directory not found: {path}")
    return name, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        action="append",
        type=parse_cohort,
        required=True,
        metavar="NAME=PATH",
        help="Direct directory containing *_1.lan files; repeat for multiple cohorts.",
    )
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trunc-level", type=int, default=15)
    parser.add_argument("--source-url")
    parser.add_argument("--source-commit")
    args = parser.parse_args()

    cohorts = dict(args.cohort)
    if len(cohorts) != len(args.cohort):
        parser.error("Cohort names must be unique")
    if not args.classification.is_file():
        parser.error(f"Classification CSV not found: {args.classification}")
    if args.trunc_level < 1:
        parser.error("--trunc-level must be positive")

    summary, manifest_path, summary_path = audit_inputs(
        cohorts=cohorts,
        classification_csv=args.classification,
        output_dir=args.output_dir,
        trunc_level=args.trunc_level,
        source_url=args.source_url,
        source_commit=args.source_commit,
    )
    print(json.dumps(summary["cohorts"], indent=2, sort_keys=True))
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
