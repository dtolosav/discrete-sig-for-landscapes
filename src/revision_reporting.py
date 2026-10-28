"""Descriptive summaries reconstructed from saved experiment outputs."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.input_audit import sha256_file
from src.revision_experiment import save_json, save_npz


def rank_correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def depth_pairs(features, depths, classes):
    """Each unordered pair once; rank correlation only, no p-values."""
    features, depths, classes = np.asarray(features), np.asarray(depths), np.asarray(classes)
    if len(features) != len(depths) or len(depths) != len(classes):
        raise ValueError("Protein alignment mismatch")
    i, j = np.triu_indices(len(features), 1)
    distances = pdist(features, metric="euclidean")
    difference = np.abs(depths[i] - depths[j])
    within = classes[i] == classes[j]
    groups = {"all": np.ones(len(i), dtype=bool), "within_classes": within,
              "between_classes": ~within}
    groups.update({f"class:{label}": within & (classes[i] == label) for label in np.unique(classes)})
    summary = {name: dict(pairs=int(mask.sum()),
                         spearman_rho=rank_correlation(distances[mask], difference[mask]))
               for name, mask in groups.items()}
    return dict(i=i, j=j, signature_distance=distances, depth_difference=difference,
                within_class=within), summary


def collect_summaries(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())["identity"]
    rows = []
    for spec in manifest["specifications"]:
        folder = directory / spec["id"]
        if not (folder / "summary.json").exists():
            raise ValueError(f"Incomplete experiment: {spec['id']}")
        summary = json.loads((folder / "summary.json").read_text())
        for relative, expected in summary["artifact_sha256"].items():
            if sha256_file(folder / relative) != expected:
                raise ValueError(f"Changed artifact: {folder / relative}")
        if summary["specification"] != spec or summary["runs"] != len(manifest["start_seeds"]):
            raise ValueError("Specification/run-count mismatch")
        scores, seconds = [], []
        for r, seeds in enumerate(manifest["start_seeds"]):
            fit = json.loads((folder / f"run-{r:03d}.json").read_text())
            if [start["seed"] for start in fit["starts"]] != seeds:
                raise ValueError("Start seeds mismatch")
            best = int(np.argmin([start["objective"] for start in fit["starts"]]))
            if (best != fit["best_start"] or fit["labels"] != fit["starts"][best]["labels"]
                    or fit["objective"] != fit["starts"][best]["objective"]):
                raise ValueError("Best start was not selected by objective")
            ari = adjusted_rand_score(manifest["classes"], fit["labels"])
            nmi = normalized_mutual_info_score(manifest["classes"], fit["labels"],
                         average_method=manifest["config"]["nmi_average_method"])
            if not np.allclose([ari, nmi], [fit["ARI"], fit["NMI"]], rtol=0, atol=1e-14):
                raise ValueError("Saved metrics differ from assignments")
            scores.append([ari, nmi])
            seconds.append(fit["seconds"])
        scores = np.array(scores)
        for k, name in enumerate(("ARI", "NMI")):
            if not np.allclose([scores[:, k].mean(), scores[:, k].std(ddof=0)],
                              [summary[name + "_mean"], summary[name + "_std"]], rtol=0, atol=1e-14):
                raise ValueError("Summary differs from per-run metrics")
        core = (summary["shared_preprocessing_seconds"] + summary["preprocessing_seconds"]
                + summary["representation_seconds"] + sum(seconds))
        if not np.isclose(core, summary["core_total_seconds"]):
            raise ValueError("Timing total mismatch")
        if not np.isclose(core + summary["warmup_seconds"], summary["total_with_setup_seconds"]):
            raise ValueError("Timing total including setup mismatch")
        rows.append({**spec, **{k: v for k, v in summary.items()
                                if k not in ("specification", "artifact_sha256", "features_sha256")},
                     "clustering_mean_seconds": float(np.mean(seconds))})
    return manifest, rows


def write_report(directory):
    directory = Path(directory)
    manifest, rows = collect_summaries(directory)
    output = directory / "reports"
    output.mkdir(exist_ok=True)
    with (output / "comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "comparisons.json", dict(scope=manifest["scope"], rows=rows,
              uncertainty="Standard deviations describe initialisation variability only",
              timing="Core totals exclude separately recorded warmup, IO and metrics"))
    with np.load(directory / "primary-dlfm/features.npz") as archive:
        pairs, depth = depth_pairs(archive["scaled"], manifest["depth"], manifest["classes"])
    save_npz(output / "depth_pairs.npz", **pairs)
    save_json(output / "depth_summary.json", dict(scope=manifest["scope"], groups=depth,
              interpretation="Descriptive association; pairs share proteins; no p-values or prediction"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = [r for r in rows if r["group"] == "primary"]
    labels = [r["method"].replace("_", " ") for r in primary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric in zip(axes, ("ARI", "NMI")):
        ax.bar(labels, [r[metric + "_mean"] for r in primary],
               yerr=[r[metric + "_std"] for r in primary], capsize=3, color="#3b718f")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=35)
    fig.suptitle(("PILOT VALIDATION — " if manifest["scope"] == "pilot" else "")
                 + "Mean ± standard deviation across repeated ten-start runs")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"primary_comparison.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    masks = [np.ones(len(pairs["i"]), dtype=bool), pairs["within_class"], ~pairs["within_class"]]
    for ax, name, mask in zip(axes, ("all", "within_classes", "between_classes"), masks):
        ax.hexbin(pairs["signature_distance"][mask], pairs["depth_difference"][mask],
                  gridsize=25, mincnt=1, cmap="Blues")
        rho = depth[name]["spearman_rho"]
        ax.set_title(name.replace("_", " ") + (f"; rho={rho:.3f}" if rho is not None else "; rho undefined"))
        ax.set_xlabel("Standardised DLFM distance")
        ax.set_ylabel("Absolute knot-depth difference")
    fig.suptitle(("PILOT VALIDATION — " if manifest["scope"] == "pilot" else "")
                 + "Descriptive unique-pair association")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"depth_association.{extension}", dpi=180)
    plt.close(fig)
    save_json(output / "artifact_sha256.json", {p.name: sha256_file(p) for p in sorted(output.iterdir())
                                               if p.is_file() and p.name != "artifact_sha256.json"})
    return rows
