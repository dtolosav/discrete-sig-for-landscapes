"""Prepare or run the approved x6 protocol, with resumable per-protein outputs.

Default scope is an 18-protein/two-run validation pilot. Full scope retains all
363 named chains and 100 runs. Both use ten starts and all approved ablations.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import shutil
import sys
from time import perf_counter
from importlib.metadata import version

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_info
from src.input_audit import load_canonical_classifications, sha256_file
from src.process_lan import load_lan
from src.revision_features import critical_grid, seed_manifest
from src.revision_experiment import specifications, bind_manifest, save_json, save_npz, run_specification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--only", nargs="+", help="Run only these specification IDs; manifest still lists all")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/revision.json")
    parser.add_argument("--input-directory", type=Path, default=ROOT /
                        "data/external/landscapes/ph-knotted-proteins/data/validation/landscapes/interpolated_x6")
    parser.add_argument("--classification", type=Path, default=ROOT / "data/external/trefoil_list.csv")
    parser.add_argument("--grid-audit", type=Path, default=ROOT.parent /
                        "results/revision-grid-audit/2026-10-28/grid_report.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.scope == "full" and not args.prepare_only and config["status"] != "ready":
        raise ValueError("Full runs require the documented driver validation gate")
    if (config["cohort"] != "x6" or not config["exclude_other"]
            or config["threads"] != 1 or config["time_multiplier"] != 1
            or config["euclidean_init"] != "k-means++"
            or config["signature_grid_levels"] != config["levels"]
            or config["level_ablation_grid"] != "keep_primary_grid"):
        raise ValueError("Unsupported protocol convention")

    start = perf_counter()
    _, labels, _ = load_canonical_classifications(args.classification)
    chains = sorted(c for c, row in labels.items()
                    if row["Representative"] and row["Representative"].lower() != "other")
    classes = sorted({labels[c]["Representative"] for c in chains})
    if len(chains) != 363 or len(labels) != 513 or len(classes) != 9:
        raise ValueError("Cohort differs from the audited x6 collection")
    all_levels = {c: load_lan(args.input_directory / f"{c}_1.lan", config["levels"]) for c in chains}
    load_seconds = perf_counter() - start
    support = max(x for lan in all_levels.values() for level in lan for x, _ in level)
    input_hashes = {c: sha256_file(args.input_directory / f"{c}_1.lan") for c in chains}
    audit = json.loads(args.grid_audit.read_text())
    if (support != config["support_L"] or input_hashes != audit["input_sha256"]
            or config["source_commit"] != audit["source_commit"]
            or sha256_file(args.classification) != audit["classification_sha256"]):
        raise ValueError("Inputs/support differ from the reconstruction audit")
    selected = chains if args.scope == "full" else sorted(
        c for label in classes for c in [c for c in chains if labels[c]["Representative"] == label][:2])
    landscapes = [all_levels[c] for c in selected]
    depth = np.array([float(labels[c]["Depth"]) for c in selected])
    derived_depth = np.array([float(labels[c]["N-tail"]) * float(labels[c]["C-tail"])
                             / float(labels[c]["Length"])**2 for c in selected])
    # The source CSV stores rounded depth values; retain them, checking counts.
    if not np.allclose(depth, derived_depth, rtol=0, atol=1e-9):
        raise ValueError("Stored depth disagrees with tail/length metadata")
    start = perf_counter()
    grid = critical_grid(landscapes, support)
    grid_seconds = perf_counter() - start
    run_seeds, seeds = seed_manifest(config["base_seed"], config["runs"], config["initialisations"])
    if args.scope == "pilot":
        run_seeds, seeds = run_seeds[:2], seeds[:2]
    specs = specifications(config)
    if args.only and not set(args.only).issubset({s["id"] for s in specs}):
        raise ValueError("Unknown specification ID")
    fruits_root = ROOT / "fruits"
    fruits_commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={fruits_root}", "-C", str(fruits_root), "rev-parse", "HEAD"], text=True).strip()
    if fruits_commit != "fd5bd3c7e4e55f6de5bcc9ff1791a64685a85c56":
        raise ValueError("FRUITS submodule differs from the verified dependency")
    scientific_config = {k: v for k, v in config.items() if k not in ("status", "full_run_gates")}
    identity = {
        "scope": args.scope, "config": scientific_config, "chains": selected,
        "classes": [labels[c]["Representative"] for c in selected],
        "depth": depth, "depth_max_rounding_discrepancy": float(np.max(np.abs(depth - derived_depth))),
        "input_sha256": input_hashes, "classification_sha256": sha256_file(args.classification),
        "source_commit": config["source_commit"], "fruits_commit": fruits_commit,
        "fruits_code_sha256": {str(p.relative_to(fruits_root)): sha256_file(p)
                                for p in sorted((fruits_root / "fruits").rglob("*.py"))},
        "grid_audit_sha256": sha256_file(args.grid_audit),
        "critical_grid_points": len(grid), "critical_grid": grid,
        "run_seeds": run_seeds, "start_seeds": seeds, "specifications": specs,
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                        [Path(__file__), ROOT / "scripts/report_revision.py", *sorted((ROOT / "src").glob("*.py"))]},
        "versions": {p: version(p) for p in
                     ("numpy", "scipy", "scikit-learn", "numba", "pysiglib", "torch")},
        "python": sys.version, "machine": platform.platform(), "host": platform.node(),
        "thread_environment": {n: os.environ[n] for n in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
        "threadpools": [{k: v for k, v in pool.items() if k != "filepath"} for pool in threadpool_info()],
        "timing_convention": "shared load/grid plus per-method evaluation, warm encoding/scaling, all starts; IO/metrics excluded",
        "resumption_convention": "retains completed-stage timings; repeated imports/loading and warmups excluded from core totals",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = bind_manifest(args.output_dir / "manifest.json", identity,
                             dict(input_loading_seconds=load_seconds, critical_grid_seconds=grid_seconds))
    # Preserve the exact uncommitted implementation as well as its hashes.
    for relative, expected in identity["code_sha256"].items():
        snapshot = args.output_dir / "source" / relative
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, snapshot)
        if sha256_file(snapshot) != expected:
            raise ValueError("Source snapshot differs from bound manifest")
    if not (args.output_dir / "grid_and_seeds.npz").exists():
        save_npz(args.output_dir / "grid_and_seeds.npz", grid=grid, run_seeds=run_seeds, start_seeds=seeds)
    with np.load(args.output_dir / "grid_and_seeds.npz") as archive:
        for key, expected in (("grid", grid), ("run_seeds", run_seeds), ("start_seeds", seeds)):
            if not np.array_equal(archive[key], expected):
                raise ValueError("Saved grid/seed archive differs from the manifest")
    print(f"Manifest: {args.scope}, {len(selected)} proteins, {len(specs)} specifications, "
          f"{len(seeds)} runs x {seeds.shape[1]} starts; grid {len(grid)}", flush=True)
    if args.prepare_only:
        return
    measurements = manifest["measurements"]
    for spec in specs:
        if args.only and spec["id"] not in args.only:
            continue
        shared = measurements["input_loading_seconds"]
        if spec["method"] not in ("flattened", "functional_l1"):
            shared += measurements["critical_grid_seconds"]
        run_specification(spec, landscapes, grid, config, identity["classes"], seeds,
                          args.output_dir / spec["id"], shared)
    finished = [s["id"] for s in specs if (args.output_dir / s["id"] / "summary.json").exists()]
    save_json(args.output_dir / "status.json", dict(completed=finished, expected=len(specs),
                                                    complete=len(finished) == len(specs)))


if __name__ == "__main__":
    main()
