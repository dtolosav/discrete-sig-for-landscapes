"""18-protein, nine-class implementation pilot: two runs, ten starts each.

This deliberately reduced validation run is not a manuscript experiment.
Usage: python scripts/revision_clustering_pilot.py --output-dir PATH
"""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
from time import perf_counter
from importlib.metadata import version

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.input_audit import load_canonical_classifications, sha256_file
from src.process_lan import load_lan
from src.revision_features import (critical_grid, evaluate, reconstruction_errors,
                                   chen_features, discrete_features, standardise, seed_manifest)
from src.integrated_chen import integrated_chen_features
from src.functional_l1 import as_functional
from src.revision_clustering import run_restarts


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/revision.json"
    config = json.loads(config_path.read_text())
    directory = ROOT / "data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_x6"
    classification = ROOT / "data/external/trefoil_list.csv"
    start = perf_counter()
    _, labels, _ = load_canonical_classifications(classification)
    chains = sorted(c for c, row in labels.items()
                    if row["Representative"] and row["Representative"].lower() != "other")
    all_levels = {c: load_lan(directory / f"{c}_1.lan", config["levels"]) for c in chains}
    support = max(x for levels in all_levels.values() for level in levels for x, _ in level)
    representatives = sorted({labels[c]["Representative"] for c in chains})
    chosen = sorted(c for representative in representatives for c in
                    [c for c in chains if labels[c]["Representative"] == representative][:2])
    if len(chains) != 363 or len(chosen) != 18 or len(representatives) != 9:
        raise ValueError("Unexpected cohort for the prescribed pilot")
    selected = [all_levels[c] for c in chosen]
    report = {"scope": "18-protein implementation pilot; no manuscript performance claim",
              "selection": "first two chains lexicographically within each named class",
              "pilot_runs": 2, "starts_per_run": config["initialisations"],
              "full_protocol_config": config,
              "config_sha256": sha256_file(config_path),
              "chains": chosen, "classes": [labels[c]["Representative"] for c in chosen],
              "input_sha256": {c: sha256_file(directory / f"{c}_1.lan") for c in chosen},
              "classification_sha256": sha256_file(classification),
              "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                              [Path(__file__), *sorted((ROOT / "src").glob("*.py"))]},
              "machine": platform.platform(),
              "thread_environment": {name: os.environ[name] for name in
                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
              "support_L": support, "input_scan_seconds": perf_counter() - start,
              "versions": {p: version(p) for p in ("numpy", "scikit-learn", "numba", "pysiglib")},
              "python": sys.version,
              "grid_scope": "union of 15-level critical points of 18 pilot proteins",
              "timing_scope": "pilot only, may include import/JIT costs, single thread",
              "methods": {}}
    start = perf_counter()
    grid = critical_grid(selected, support)
    values = evaluate(selected, grid)
    report["shared_critical_preprocessing_seconds"] = perf_counter() - start
    report["critical_grid_points"] = len(grid)
    report["critical_reconstruction_error"] = float(reconstruction_errors(selected, grid).max())
    report["uniform_reconstruction_errors"] = {
        str(n): reconstruction_errors(selected, np.linspace(0, support, n)).tolist()
        for n in (512, 1024, 2048)}
    seeds, start_seeds = seed_manifest(config["base_seed"], config["runs"], config["initialisations"])
    archive = {"chains": np.array(chosen), "run_seeds": seeds,
               "start_seeds": start_seeds, "critical_grid": grid}
    pilot_starts = start_seeds[:2]
    methods = ("dlfm", "chen", "time_chen", "integrated_chen", "flattened", "functional_l1")
    for method in methods:
        print(f"Starting {method}", flush=True)
        start = perf_counter()
        method_preprocessing = 0.
        if method == "functional_l1":
            data = as_functional(selected, support)
            method_preprocessing = perf_counter() - start
            encoding = 0.
            mode = "functional_l1"
            details = {"geometry": "unsquared integrated L1, pointwise median centres"}
        else:
            mode = "euclidean"
            if method == "dlfm":
                features = discrete_features(values, config["weight"])
            elif method == "integrated_chen":
                features = integrated_chen_features(values, grid, config["weight"])
            elif method in ("chen", "time_chen"):
                features = chen_features(values, grid, config["weight"], method == "time_chen")
            else:
                sampled = evaluate(selected, np.linspace(0, support, config["flattened_grid_points"]))
                method_preprocessing = perf_counter() - start
                start = perf_counter()
                features = sampled.reshape(len(selected), -1)
            data, scaler, keep = standardise(features)
            encoding = perf_counter() - start
            details = {"raw_shape": list(features.shape), "retained_shape": list(data.shape)}
            archive.update({f"{method}_raw": features, f"{method}_scaled": data,
                            f"{method}_keep": keep, f"{method}_mean": scaler.mean_,
                            f"{method}_scale": scaler.scale_})
        fits = run_restarts(data, mode, config["clusters"], pilot_starts, max_iter=config["max_iter"])
        for fit in fits:
            fit["ARI"] = adjusted_rand_score(report["classes"], fit["labels"])
            fit["NMI"] = normalized_mutual_info_score(report["classes"], fit["labels"],
                                                      average_method=config["nmi_average_method"])
        details.update({"method_preprocessing_seconds": method_preprocessing,
                        "representation_seconds": encoding, "runs": fits})
        report["methods"][method] = details
        (args.output_dir / "clustering_report.json").write_text(
            json.dumps(report, default=json_value, indent=2) + "\n")
        np.savez_compressed(args.output_dir / "clustering_features.npz", **archive)
        print(f"Finished {method}; raw/retained {details.get('raw_shape')}/{details.get('retained_shape')}; "
              f"clustering seconds {[round(f['seconds'], 3) for f in fits]}; "
              f"iteration caps {sum(s['hit_iteration_limit'] for f in fits for s in f['starts'])}", flush=True)


if __name__ == "__main__":
    main()
