"""Assess uniform-grid reconstruction on x6 inputs, without clustering.

Errors concern the supplied PWL landscapes through level 15, not barcode
reconstruction or preservation of standardised-feature distances.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.input_audit import load_canonical_classifications, sha256_file
from src.process_lan import load_lan
from src.revision_features import critical_grid, evaluate, reconstruction_errors
from src.functional_l1 import as_functional, l1_distance


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
    if len(chains) != 363:
        raise ValueError("Expected 363 named x6 inputs")
    levels = [load_lan(directory / f"{c}_1.lan", config["levels"]) for c in chains]
    support = max(x for landscape in levels for level in landscape for x, _ in level)
    functional = as_functional(levels, support)
    mass = np.array([sum(np.trapz(y, x) for x, y in landscape) for landscape in functional])
    if np.any(mass <= 0):
        raise ValueError("Relative L1 errors require positive landscape mass")
    report = {
        "scope": "input reconstruction audit only; no signatures or clustering",
        "config": config, "support_L": support, "levels": config["levels"],
        "critical_grid_points_with_endpoints": len(critical_grid(levels, support)),
        "chains": chains, "landscape_L1_mass": mass.tolist(),
        "source_commit": config["source_commit"],
        "classification_sha256": sha256_file(classification),
        "input_sha256": {c: sha256_file(directory / f"{c}_1.lan") for c in chains},
        "config_sha256": sha256_file(config_path),
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                        [Path(__file__), *sorted((ROOT / "src").glob("*.py"))]},
        "grids": {},
    }
    for n in sorted({config["flattened_grid_points"], *config["ablation_grid_points"]}):
        grid = np.linspace(0, support, n)
        sampled = evaluate(levels, grid)
        sup = reconstruction_errors(levels, grid)
        l1 = np.array([sum(l1_distance(curve, (grid, sampled[i, r]))
                           for r, curve in enumerate(landscape))
                       for i, landscape in enumerate(functional)])
        relative = l1 / mass
        summary = {"max_sup_error": float(sup.max()),
                   "median_relative_L1_error": float(np.median(relative)),
                   "p95_relative_L1_error": float(np.quantile(relative, .95)),
                   "max_relative_L1_error": float(relative.max()),
                   "largest_sup_error_chain": chains[int(sup.argmax())],
                   "largest_relative_L1_error_chain": chains[int(relative.argmax())]}
        report["grids"][str(n)] = {"points": n, "spacing": support / (n - 1),
            "raw_dimension": n * config["levels"], "summary": summary,
            "sup_error": sup.tolist(), "L1_error": l1.tolist(),
            "relative_L1_error": relative.tolist()}
        print(n, summary, flush=True)
    (args.output_dir / "grid_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
