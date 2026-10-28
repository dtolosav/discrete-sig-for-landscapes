"""Reproducible three-protein x6 feature pilot; never a full experiment.

Run from any directory: python scripts/revision_pilot.py --output-dir PATH
Outputs contain implementation checks, not manuscript-facing performance scores.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

# Fix numerical thread pools before importing scientific dependencies.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.input_audit import load_canonical_classifications, sha256_file
from src.process_lan import load_lan
from src.revision_features import (critical_grid, evaluate, reconstruction_errors,
                                   discrete_features, chen_features, standardise,
                                   seed_manifest)
from src.functional_l1 import median_curve, l1_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/revision.json"
    config = json.loads(config_path.read_text())
    directory = ROOT / "data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_x6"
    classification = ROOT / "data/external/trefoil_list.csv"
    _, labels, _ = load_canonical_classifications(classification)
    chains = sorted(c for c, row in labels.items()
                    if row["Representative"] and row["Representative"].lower() != "other")
    if len(chains) != 363:
        raise ValueError("Expected the audited 363-protein named cohort")
    start = perf_counter()
    # Input-only scan fixes a common support for the whole intended cohort.
    landscapes = [load_lan(directory / f"{c}_1.lan", config["levels"]) for c in chains]
    support = max(x for levels in landscapes for level in levels for x, _ in level)
    report = {"scope": "three-protein implementation pilot, no clustering conclusions",
              "config": config, "config_sha256": sha256_file(config_path),
              "classification_sha256": sha256_file(classification),
              "python": sys.version, "numpy": np.__version__,
              "support_L": support, "cohort_input_scan_seconds": perf_counter() - start,
              "selection": "first three named chain identifiers in lexicographic order",
              "chains": chains[:3], "input_sha256": {
                  c: sha256_file(directory / f"{c}_1.lan") for c in chains[:3]},
              "grid_scope": "critical-point union of three pilot proteins, not the full cohort",
              "timing_scope": "single cold pilot execution; not benchmark results"}
    selected = landscapes[:3]
    grid = critical_grid(selected, support)
    report["critical_grid_points"] = len(grid)
    values = evaluate(selected, grid)
    assert np.max(reconstruction_errors(selected, grid)) < 1e-10
    report["uniform_reconstruction"] = {}
    for n in [512, 1024, 2048]:
        uniform = np.linspace(0, support, n)
        report["uniform_reconstruction"][str(n)] = reconstruction_errors(selected, uniform).tolist()
    archive = {"chains": np.array(chains[:3]), "critical_grid": grid}
    report["features"] = {}
    for method in ("dlfm", "chen", "time_chen", "flattened"):
        start = perf_counter()
        if method == "dlfm":
            features = discrete_features(values, config["weight"])
        elif method in ("chen", "time_chen"):
            features = chen_features(values, grid, config["weight"], method == "time_chen")
            d = config["levels"] + (method == "time_chen")
            assert features.shape[1] == sum(d**w for w in range(1, config["weight"]+1))
        else:
            features = evaluate(selected, np.linspace(0, support, config["flattened_grid_points"]))
            features = features.reshape(len(selected), -1)
        scaled, scaler, keep = standardise(features)
        archive.update({f"{method}_raw": features, f"{method}_scaled": scaled,
                        f"{method}_keep": keep, f"{method}_mean": scaler.mean_,
                        f"{method}_scale": scaler.scale_})
        report["features"][method] = {"raw_shape": list(features.shape),
                                      "retained_shape": list(scaled.shape),
                                      "seconds_including_scaling": perf_counter()-start}
        print(method, report["features"][method], flush=True)
    curves = [tuple(np.asarray(lan[0]).T) for lan in selected]
    start = perf_counter()
    centre = median_curve(curves)
    report["l1_first_level_check"] = {
        "seconds": perf_counter()-start,
        "centre_knots": len(centre[0]),
        "median_objective": sum(l1_distance(c, centre) for c in curves),
        "first_member_objective": sum(l1_distance(c, curves[0]) for c in curves)}
    assert report["l1_first_level_check"]["median_objective"] <= report["l1_first_level_check"]["first_member_objective"] + 1e-9
    archive["run_seeds"], archive["initialisation_seeds"] = seed_manifest(
        config["base_seed"], config["runs"], config["initialisations"])
    np.savez_compressed(args.output_dir / "pilot_features.npz", **archive)
    (args.output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["uniform_reconstruction"]), flush=True)


if __name__ == "__main__":
    main()
