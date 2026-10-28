"""Bounded resource checks, not the full comparative experiment.

Compute signatures for one protein on the full x6 grid and one ten-start L1
pilot on at most eight proteins per class. No ARI/NMI or significance tests.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.input_audit import load_canonical_classifications, sha256_file
from src.process_lan import load_lan
from src.revision_features import critical_grid, evaluate, discrete_features, chen_features, seed_manifest
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
    classification = ROOT / "data/external/trefoil_list.csv"
    directory = ROOT / "data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_x6"
    _, labels, _ = load_canonical_classifications(classification)
    chains = sorted(c for c, row in labels.items()
                    if row["Representative"] and row["Representative"].lower() != "other")
    levels = {c: load_lan(directory / f"{c}_1.lan", config["levels"]) for c in chains}
    support = max(x for lan in levels.values() for level in lan for x, _ in level)
    grid = critical_grid(list(levels.values()), support)
    selected = max(chains, key=lambda c: sum(map(len, levels[c])))
    values = evaluate([levels[selected]], grid)
    own_grid = critical_grid([levels[selected]], support)
    own_values = evaluate([levels[selected]], own_grid)
    report = {
        "scope": __doc__, "config": config, "support_L": support,
        "full_grid_points": len(grid), "selected_chain": selected,
        "selection": "largest total retained knot count; lexicographic first on ties",
        "own_grid_points": len(own_grid), "sample_array_bytes": values.nbytes,
        "hypothetical_full_cohort_array_bytes": len(chains) * values.nbytes,
        "input_sha256": {c: sha256_file(directory / f"{c}_1.lan") for c in chains},
        "classification_sha256": sha256_file(classification),
        "config_sha256": sha256_file(config_path),
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                        [Path(__file__), *sorted((ROOT / "src").glob("*.py"))]},
        "timing_scope": "single thread, one warm execution; excludes IO and tiny warmup",
        "methods": {},
    }

    def save():
        report["process_peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output_dir / "feasibility_report.json").write_text(
            json.dumps(report, default=json_value, indent=2) + "\n")

    methods = {
        "dlfm": lambda v, g: discrete_features(v, config["weight"]),
        "chen": lambda v, g: chen_features(v, g, config["weight"]),
        "time_chen": lambda v, g: chen_features(v, g, config["weight"], True),
        "integrated_chen": lambda v, g: integrated_chen_features(v, g, config["weight"]),
    }
    for name, encode in methods.items():
        print(f"Single full-grid protein: {name}", flush=True)
        encode(values[:, :, [0, len(grid) // 2, -1]], grid[[0, len(grid) // 2, -1]])
        start = perf_counter()
        features = encode(values, grid)
        seconds = perf_counter() - start
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite signature")
        details = {"seconds": seconds, "raw_dimension": features.shape[1]}
        if name != "dlfm":
            reference = encode(own_values, own_grid)
            details["own_grid_max_absolute_discrepancy"] = float(np.max(np.abs(features - reference)))
            details["relative_discrepancy_to_largest_coordinate"] = float(
                np.max(np.abs(features - reference)) / max(1., np.max(np.abs(reference))))
        report["methods"][name] = details
        save()
        print(name, details, flush=True)

    representatives = sorted({labels[c]["Representative"] for c in chains})
    subset = sorted(c for representative in representatives for c in
                    [c for c in chains if labels[c]["Representative"] == representative][:8])
    data = as_functional([levels[c] for c in subset], support)
    _, seeds = seed_manifest(config["base_seed"], config["runs"], config["initialisations"])
    print(f"Functional L1: {len(subset)} proteins, one run with ten starts", flush=True)
    fits = run_restarts(data, "functional_l1", config["clusters"], seeds[:1], config["max_iter"])
    report["functional_l1"] = {"chains": subset, "selection": "first up to eight per named class",
                                "runs": fits}
    save()
    print("L1 seconds:", fits[0]["seconds"], "iteration caps:",
          sum(s["hit_iteration_limit"] for s in fits[0]["starts"]), flush=True)


if __name__ == "__main__":
    main()
