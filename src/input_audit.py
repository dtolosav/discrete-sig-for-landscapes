"""Audit landscape inputs and reconcile them with protein classifications."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.process_lan import load_lan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chain_id_from_landscape(path: Path) -> str:
    suffix = "_1.lan"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected an H1 landscape filename ending in {suffix}: {path.name}")
    return path.name[: -len(suffix)]


def load_canonical_classifications(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            cleaned = {
                key: (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            cleaned["_complete"] = all(row.get(key) is not None for key in reader.fieldnames)
            rows.append(cleaned)

    required = {"Chain", "Representative"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Classification CSV must contain {sorted(required)}: {path}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["Chain"]].append(row)

    canonical = {}
    duplicate_chains = {}
    for chain, chain_rows in grouped.items():
        if not chain:
            raise ValueError(f"Classification CSV contains an empty Chain value: {path}")

        complete_rows = [row for row in chain_rows if row["_complete"]]
        if not complete_rows:
            raise ValueError(f"No complete classification row for chain {chain}")

        metadata = {
            tuple(
                (key, value)
                for key, value in row.items()
                if key not in {"Representative", "_complete"}
            )
            for row in complete_rows
        }
        if len(metadata) != 1:
            raise ValueError(f"Conflicting metadata for duplicated chain {chain}")

        representatives = {
            row["Representative"] for row in complete_rows if row["Representative"]
        }
        if len(representatives) > 1:
            raise ValueError(f"Conflicting representative labels for duplicated chain {chain}")

        selected = dict(
            next(
                (row for row in complete_rows if row["Representative"]),
                complete_rows[0],
            )
        )
        selected["Representative"] = next(iter(representatives), "")
        selected.pop("_complete")
        canonical[chain] = selected
        if len(chain_rows) > 1:
            duplicate_chains[chain] = len(chain_rows)

    return rows, canonical, duplicate_chains


def audit_inputs(
    cohorts: dict[str, Path],
    classification_csv: Path,
    output_dir: Path,
    trunc_level: int = 15,
    source_url: str | None = None,
    source_commit: str | None = None,
):
    raw_labels, labels, duplicate_chains = load_canonical_classifications(classification_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    cohort_summaries = {}
    cohort_chain_sets = {}

    for cohort, directory in sorted(cohorts.items()):
        files = sorted(directory.glob("*_1.lan"))
        if not files:
            raise ValueError(f"No direct H1 landscape files found in {directory}")

        cohort_chains = set()
        included_grid = set()
        all_grid = set()
        class_counts = Counter()
        unmatched = []
        invalid = []

        for path in files:
            chain = chain_id_from_landscape(path)
            if chain in cohort_chains:
                raise ValueError(f"Duplicate landscape chain in {cohort}: {chain}")
            cohort_chains.add(chain)

            label = labels.get(chain)
            representative = label["Representative"] if label else ""
            if label is None:
                label_status = "unmatched"
                unmatched.append(chain)
            elif not representative:
                label_status = "blank"
            elif representative == "Other":
                label_status = "other"
            else:
                label_status = "named"
                class_counts[representative] += 1

            try:
                levels = load_lan(path, trunc_level)
                points = sum(len(level) for level in levels)
                nonempty_levels = sum(bool(level) for level in levels)
                file_grid = {x for level in levels for x, _ in level}
                all_grid.update(file_grid)
                if label_status == "named":
                    included_grid.update(file_grid)
                parse_status = "ok"
            except (OSError, ValueError) as exc:
                points = 0
                nonempty_levels = 0
                parse_status = f"error: {exc}"
                invalid.append(path.name)

            manifest_rows.append(
                {
                    "cohort": cohort,
                    "chain": chain,
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "classification_row_count": len(
                        [row for row in raw_labels if row["Chain"].strip() == chain]
                    ),
                    "representative": representative,
                    "label_status": label_status,
                    "included_named_cohort": label_status == "named",
                    "parse_status": parse_status,
                    "nonempty_levels_through_truncation": nonempty_levels,
                    "critical_points_through_truncation": points,
                }
            )

        cohort_chain_sets[cohort] = cohort_chains
        cohort_summaries[cohort] = {
            "directory": str(directory.resolve()),
            "h1_landscape_files": len(files),
            "matched_unique_chains": len(cohort_chains & labels.keys()),
            "unmatched_landscape_chains": sorted(unmatched),
            "classification_chains_without_landscape": sorted(labels.keys() - cohort_chains),
            "named_cohort_files": sum(count for count in class_counts.values()),
            "other_files": sum(
                row["cohort"] == cohort and row["label_status"] == "other"
                for row in manifest_rows
            ),
            "blank_label_files": sum(
                row["cohort"] == cohort and row["label_status"] == "blank"
                for row in manifest_rows
            ),
            "invalid_landscape_files": invalid,
            "class_counts_named_cohort": dict(sorted(class_counts.items())),
            "global_critical_grid_size_all_files": len(all_grid),
            "global_critical_grid_size_named_cohort": len(included_grid),
            "flattened_features_named_cohort": trunc_level * len(included_grid),
        }

    cohort_names = sorted(cohort_chain_sets)
    comparisons = {}
    for index, left in enumerate(cohort_names):
        for right in cohort_names[index + 1 :]:
            comparisons[f"{left}_vs_{right}"] = {
                "only_in_left": sorted(cohort_chain_sets[left] - cohort_chain_sets[right]),
                "only_in_right": sorted(cohort_chain_sets[right] - cohort_chain_sets[left]),
            }

    canonical_counts = Counter(
        "blank" if not row["Representative"] else row["Representative"]
        for row in labels.values()
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "source_commit": source_commit,
        "classification_csv": str(classification_csv.resolve()),
        "classification_sha256": sha256_file(classification_csv),
        "classification_rows": len(raw_labels),
        "classification_incomplete_rows": sum(not row["_complete"] for row in raw_labels),
        "classification_unique_chains": len(labels),
        "classification_duplicate_extra_rows": len(raw_labels) - len(labels),
        "classification_duplicate_chains": dict(sorted(duplicate_chains.items())),
        "canonical_named_chains": sum(
            count for name, count in canonical_counts.items() if name not in {"blank", "Other"}
        ),
        "canonical_other_chains": canonical_counts["Other"],
        "canonical_blank_chains": canonical_counts["blank"],
        "truncation_level_for_parse_and_grid_audit": trunc_level,
        "cohorts": cohort_summaries,
        "cohort_chain_comparisons": comparisons,
    }

    manifest_path = output_dir / "landscape_input_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_dir / "landscape_input_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary, manifest_path, summary_path
