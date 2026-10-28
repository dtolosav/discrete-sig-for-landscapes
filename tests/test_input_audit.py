import csv
from pathlib import Path

import pytest

from src.input_audit import chain_id_from_landscape, load_canonical_classifications


def write_classification(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Chain", "Depth", "Length", "N-tail", "C-tail", "Representative"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_chain_id_from_h1_landscape():
    assert chain_id_from_landscape(Path("3kzk_A_1.lan")) == "3kzk_A"
    with pytest.raises(ValueError):
        chain_id_from_landscape(Path("3kzk_A_0.lan"))


def test_duplicate_blank_label_is_canonicalised(tmp_path):
    classification = tmp_path / "labels.csv"
    base = {
        "Chain": "1abc_A",
        "Depth": "0.1",
        "Length": "100",
        "N-tail": "10",
        "C-tail": "10",
    }
    write_classification(
        classification,
        [
            {**base, "Representative": "2xyz_A"},
            {**base, "Representative": ""},
        ],
    )

    raw, canonical, duplicates = load_canonical_classifications(classification)

    assert len(raw) == 2
    assert canonical["1abc_A"]["Representative"] == "2xyz_A"
    assert duplicates == {"1abc_A": 2}


def test_malformed_duplicate_is_ignored_when_complete_row_exists(tmp_path):
    classification = tmp_path / "labels.csv"
    classification.write_text(
        "Chain,Depth,Length,N-tail,C-tail,Representative\n"
        "1abc_A,0.1,100,10,10,2xyz_A\n"
        "1abc_A,0.1,100,10,2xyz_A\n",
        encoding="utf-8",
    )

    raw, canonical, duplicates = load_canonical_classifications(classification)

    assert raw[1]["_complete"] is False
    assert canonical["1abc_A"]["C-tail"] == "10"
    assert canonical["1abc_A"]["Representative"] == "2xyz_A"
    assert duplicates == {"1abc_A": 2}


def test_conflicting_duplicate_labels_are_rejected(tmp_path):
    classification = tmp_path / "labels.csv"
    base = {
        "Chain": "1abc_A",
        "Depth": "0.1",
        "Length": "100",
        "N-tail": "10",
        "C-tail": "10",
    }
    write_classification(
        classification,
        [
            {**base, "Representative": "2xyz_A"},
            {**base, "Representative": "3xyz_A"},
        ],
    )

    with pytest.raises(ValueError, match="Conflicting representative"):
        load_canonical_classifications(classification)
