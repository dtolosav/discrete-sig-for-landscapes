import json
from pathlib import Path
import numpy as np
import pytest
from src.revision_experiment import (METHODS, specifications, bind_manifest,
                                      run_specification, encode)
from src.revision_features import evaluate, standardise

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_has_only_prescribed_one_factor_ablations():
    config = json.loads((ROOT / "configs/revision.json").read_text())
    specs = specifications(config)
    assert len(specs) == 24 and len({s["id"] for s in specs}) == 24
    for s in specs:
        changes = sum((s["levels"] != 15, s["weight"] != 3, s["points"] != 1024))
        assert changes == (0 if s["group"] == "primary" else 1)
        if s["group"].startswith("grid"):
            assert s["method"] == "flattened"
        if s["group"].startswith("weight"):
            assert s["method"] in METHODS[:4]


def test_manifest_rejects_changed_inputs_or_settings(tmp_path):
    path = tmp_path / "manifest.json"
    identity = {"chains": ["a", "b"], "seeds": np.array([[1, 2]])}
    first = bind_manifest(path, identity, {"seconds": 1.})
    resumed = bind_manifest(path, identity, {"seconds": 9.})
    assert first["measurements"] == resumed["measurements"]
    with pytest.raises(ValueError, match="Manifest differs"):
        bind_manifest(path, {**identity, "chains": ["b", "a"]}, {})


@pytest.mark.parametrize("method", METHODS)
def test_streamed_experiment_matches_batch_and_resumes_without_refitting(tmp_path, monkeypatch, method):
    config = json.loads((ROOT / "configs/revision.json").read_text())
    config.update(support_L=2., clusters=2)
    grid = np.array([0., .4, .9, 1.3, 2.])
    rng = np.random.RandomState(23)
    landscapes = [[list(zip(grid, np.r_[0., rng.uniform(.05, 1., 3), 0.]))
                   for _ in range(2)] for _ in range(6)]
    spec = dict(id="test-" + method, group="test", method=method, levels=2, weight=2, points=9)
    seeds = np.array([[42, 43], [44, 45]])
    result = run_specification(spec, landscapes, grid, config, [0, 0, 0, 1, 1, 1], seeds,
                               tmp_path, 1.25)
    assert result["runs"] == 2 and result["initialisations"] == 2
    assert result["core_total_seconds"] == pytest.approx(
        1.25 + result["preprocessing_seconds"] + result["representation_seconds"]
        + result["clustering_seconds"])
    if method != "functional_l1":
        selected_grid = np.linspace(0, 2, 9) if method == "flattened" else grid
        batch = encode(evaluate(landscapes, selected_grid), selected_grid, spec)
        scaled, _, keep = standardise(batch)
        with np.load(tmp_path / "features.npz") as stored:
            np.testing.assert_allclose(stored["raw"], batch, atol=1e-13)
            np.testing.assert_array_equal(stored["keep"], keep)
            np.testing.assert_allclose(stored["scaled"], scaled, atol=1e-12)
    monkeypatch.setattr("src.revision_experiment.run_restarts",
                        lambda *a, **k: pytest.fail("Unexpected repeated fitting"))
    resumed = run_specification(spec, landscapes, grid, config, [0, 0, 0, 1, 1, 1], seeds,
                                tmp_path, 1.25)
    assert resumed == json.loads((tmp_path / "summary.json").read_text())
    # Simulate interruption after one completed run, without deleting features.
    (tmp_path / "summary.json").unlink()
    # Both run checkpoints remain; reconstruction must not invoke fitting.
    rebuilt = run_specification(spec, landscapes, grid, config, [0, 0, 0, 1, 1, 1], seeds,
                                tmp_path, 1.25)
    assert rebuilt["ARI_mean"] == result["ARI_mean"]
    (tmp_path / "centres-000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed output artifact"):
        run_specification(spec, landscapes, grid, config, [0, 0, 0, 1, 1, 1], seeds,
                          tmp_path, 1.25)
