"""Small-batch encoding and resumable, objective-selected experiments."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.input_audit import sha256_file
from src.revision_features import evaluate, chen_features, discrete_features, standardise
from src.integrated_chen import integrated_chen_features
from src.functional_l1 import as_functional
from src.revision_clustering import run_restarts

METHODS = ("dlfm", "chen", "time_chen", "integrated_chen", "flattened", "functional_l1")


def specifications(config):
    """Primary six methods and approved one-factor ablations only."""
    specs = []

    def add(group, methods, levels=None, weight=None, points=None):
        for method in methods:
            specs.append(dict(id=f"{group}-{method}", group=group, method=method,
                              levels=levels or config["levels"],
                              weight=weight or config["weight"],
                              points=points or config["flattened_grid_points"]))
    add("primary", METHODS)
    for d in config["ablation_levels"]:
        add(f"levels-{d}", METHODS, levels=d)
    for k in config["ablation_weights"]:
        add(f"weight-{k}", METHODS[:4], weight=k)
    for n in config["ablation_grid_points"]:
        add(f"grid-{n}", ("flattened",), points=n)
    return specs


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, default=json_value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def bind_manifest(path, identity, measurements):
    """Never mix outputs after changing inputs, code, settings or environment."""
    if path.exists():
        existing = json.loads(path.read_text())
        # Normalise numpy types exactly as in the saved JSON.
        normalised = json.loads(json.dumps(identity, default=json_value))
        if existing["identity"] != normalised:
            raise ValueError("Manifest differs: use a new output directory")
        return existing
    manifest = {"identity": identity, "measurements": measurements}
    save_json(path, manifest)
    return manifest


def encode(values, grid, spec):
    method, weight = spec["method"], spec["weight"]
    if method == "dlfm":
        return discrete_features(values, weight)
    if method == "integrated_chen":
        return integrated_chen_features(values, grid, weight)
    if method in ("chen", "time_chen"):
        return chen_features(values, grid, weight, method == "time_chen")
    if method == "flattened":
        return values.reshape(len(values), -1)
    raise ValueError(method)


def run_specification(spec, landscapes, grid, config, classes, seeds, output, shared_seconds):
    """Checkpoint every protein and every complete ten-start clustering run.

    Only one protein's sampled landscape is resident during encoding. All raw
    feature rows are assembled before fitting one scaler for the whole cohort.
    IO and class-agreement scoring are outside measured numerical stages.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for relative, expected in summary["artifact_sha256"].items():
            if sha256_file(output / relative) != expected:
                raise ValueError(f"Changed output artifact: {output / relative}")
        return summary
    method = spec["method"]
    ready_path = output / "representation.json"
    artifacts = []
    if ready_path.exists():
        timing = json.loads(ready_path.read_text())
    else:
        timing = {}

    start = perf_counter()
    retained = [lan[:spec["levels"]] for lan in landscapes]
    selection_seconds = perf_counter() - start
    if method == "functional_l1":
        start = perf_counter()
        data = as_functional(retained, config["support_L"])
        if not timing:
            timing = dict(preprocessing_seconds=selection_seconds + perf_counter() - start,
                          representation_seconds=0., warmup_seconds=0.,
                          raw_dimension=None, retained_dimension=None)
            save_json(ready_path, timing)
        mode = "functional_l1"
    else:
        mode = "euclidean"
        feature_path = output / "features.npz"
        if not timing:
            start = perf_counter()
            sample_grid = (np.linspace(0., config["support_L"], spec["points"])
                           if method == "flattened" else grid)
            grid_seconds = perf_counter() - start
            # Tiny representative warmup makes the steady-state timing explicit.
            start = perf_counter()
            tiny_grid = np.array([0., config["support_L"] / 2, config["support_L"]])
            encode(evaluate(retained[:1], tiny_grid), tiny_grid, spec)
            warmup = perf_counter() - start
            rows_dir = output / "rows"
            rows_dir.mkdir(exist_ok=True)
            rows, prep_seconds, representation_seconds = [], selection_seconds + grid_seconds, 0.
            for i, landscape in enumerate(retained):
                path = rows_dir / f"{i:04d}.npz"
                if not path.exists():
                    start = perf_counter()
                    values = evaluate([landscape], sample_grid)
                    prep = perf_counter() - start
                    start = perf_counter()
                    row = encode(values, sample_grid, spec)[0]
                    elapsed = perf_counter() - start
                    if not np.isfinite(row).all():
                        raise ValueError(f"Nonfinite features for row {i}")
                    save_npz(path, row=row, preprocessing_seconds=prep, representation_seconds=elapsed)
                with np.load(path) as stored:
                    rows.append(stored["row"])
                    prep_seconds += float(stored["preprocessing_seconds"])
                    representation_seconds += float(stored["representation_seconds"])
                if (i + 1) % 20 == 0 or i + 1 == len(retained):
                    print(f"{spec['id']}: encoded {i + 1}/{len(retained)}", flush=True)
            start = perf_counter()
            raw = np.stack(rows)
            data, scaler, keep = standardise(raw)
            representation_seconds += perf_counter() - start
            save_npz(feature_path, raw=raw, scaled=data, keep=keep,
                     mean=scaler.mean_, scale=scaler.scale_)
            timing = dict(preprocessing_seconds=prep_seconds,
                          representation_seconds=representation_seconds, warmup_seconds=warmup,
                          raw_dimension=raw.shape[1], retained_dimension=data.shape[1],
                          features_sha256=sha256_file(feature_path))
            save_json(ready_path, timing)
        if sha256_file(feature_path) != timing["features_sha256"]:
            raise ValueError("Changed feature archive")
        with np.load(feature_path) as stored:
            data = stored["scaled"]
        artifacts.append(feature_path)

    runs = []
    for run_index, start_seeds in enumerate(seeds):
        run_path = output / f"run-{run_index:03d}.json"
        centres_path = output / f"centres-{run_index:03d}.npz"
        if run_path.exists():
            fit = json.loads(run_path.read_text())
            if sha256_file(centres_path) != fit["centres_sha256"]:
                raise ValueError("Changed centre archive")
        else:
            fit = run_restarts(data, mode, config["clusters"], [start_seeds],
                               config["max_iter"], config["euclidean_tol"])[0]
            centres = fit.pop("centres")
            if mode == "euclidean":
                save_npz(centres_path, centres=centres)
            else:
                save_npz(centres_path, **{f"c{c}_level{r}_{name}": values
                    for c, landscape in enumerate(centres) for r, curve in enumerate(landscape)
                    for name, values in zip(("knots", "values"), curve)})
            fit["centres_sha256"] = sha256_file(centres_path)
            fit["ARI"] = adjusted_rand_score(classes, fit["labels"])
            fit["NMI"] = normalized_mutual_info_score(classes, fit["labels"],
                                                       average_method=config["nmi_average_method"])
            save_json(run_path, fit)
            print(f"{spec['id']}: run {run_index + 1}/{len(seeds)} complete", flush=True)
        runs.append(fit)
        artifacts.extend([run_path, centres_path])
    artifacts.append(ready_path)
    summary = {"specification": spec, "runs": len(runs), "initialisations": len(seeds[0]),
               **timing, "shared_preprocessing_seconds": shared_seconds,
               "clustering_seconds": sum(fit["seconds"] for fit in runs),
               "iteration_limit_starts": sum(s["hit_iteration_limit"] for f in runs for s in f["starts"]),
               "occupied_clusters_min": min(s["occupied_clusters"] for f in runs for s in f["starts"]),
               "standard_deviation_ddof": 0}
    for metric in ("ARI", "NMI"):
        values = [fit[metric] for fit in runs]
        summary[metric + "_mean"] = float(np.mean(values))
        summary[metric + "_std"] = float(np.std(values, ddof=0))
    summary["core_total_seconds"] = (shared_seconds + timing["preprocessing_seconds"]
                                     + timing["representation_seconds"] + summary["clustering_seconds"])
    summary["total_with_setup_seconds"] = summary["core_total_seconds"] + timing["warmup_seconds"]
    summary["artifact_sha256"] = {str(p.relative_to(output)): sha256_file(p) for p in artifacts}
    save_json(summary_path, summary)
    return summary
