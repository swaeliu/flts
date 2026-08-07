"""Unit tests for atomic persistence, manifest validation, and resume logic."""

import json
import os

import pytest
import torch

from gvendi.candidates import CandidateEvaluation, CandidateSpec
from gvendi.config import GVendiConfig
from gvendi.grad_vendi import SelectionCandidate
from gvendi.manifest import (
    GVendiPaths,
    atomic_torch_save,
    atomic_write_json,
    build_selection_manifest,
    load_candidate_pool,
    save_candidate_pool,
    selected_client_row,
    try_load_selection_manifest,
    validate_selection_manifest,
)


class TestAtomicWrites:
    def test_json_roundtrip_no_tmp_left(self, tmp_path):
        path = tmp_path / "x.json"
        atomic_write_json(str(path), {"a": 1})
        with open(path) as f:
            assert json.load(f) == {"a": 1}
        assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]

    def test_json_overwrites(self, tmp_path):
        path = tmp_path / "x.json"
        atomic_write_json(str(path), {"a": 1})
        atomic_write_json(str(path), {"a": 2})
        with open(path) as f:
            assert json.load(f) == {"a": 2}

    def test_torch_save_roundtrip(self, tmp_path):
        path = tmp_path / "t.pt"
        t = torch.randn(3, 4)
        atomic_torch_save(str(path), {"t": t})
        loaded = torch.load(str(path), weights_only=False)
        assert torch.equal(loaded["t"], t)
        assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]


class TestCandidatePool:
    def test_roundtrip(self, tmp_path):
        cfg = GVendiConfig()
        specs = [
            CandidateSpec("c0", 0, {"family": "trend_seasonal", "trend_slope": 0.1},
                          "perturb", 900_000, [2]),
            CandidateSpec("c1", 7, {"family": "sarima_like", "trend_slope": -0.1},
                          "new_regime", 901_000),
        ]
        path = str(tmp_path / "pool.json")
        save_candidate_pool(path, specs, cfg)
        loaded = [CandidateSpec.from_dict(d) for d in load_candidate_pool(path)]
        assert [s.to_dict() for s in loaded] == [s.to_dict() for s in specs]


def _make_manifest(cfg, prev_r=2, prev_v=2, new_v=3, method="h_vendi"):
    rows = []
    for r in range(prev_r):
        rows.append(_row(f"e{r}", r, prev_v))
    for v in range(new_v):
        rows.append(_row(f"n{v}", prev_r, v))
    return build_selection_manifest(
        cfg, method, f"{prev_r}x{prev_v}", f"{prev_r + 1}x{new_v}",
        "/ckpt", rows,
    )


def _row(cid, regime, variant):
    return {
        "client_key": f"regime_{regime:02d}_variant_{variant:02d}",
        "candidate_id": cid,
        "regime_index": regime,
        "variant_index": variant,
        "generator_parameters": {"family": "trend_seasonal"},
        "data_seed": 900_000,
    }


class TestSelectionManifest:
    def test_valid_manifest_passes(self):
        cfg = GVendiConfig()
        m = _make_manifest(cfg)
        validate_selection_manifest(m, "h_vendi", 2, 2, 3, 3)

    def test_wrong_method_rejected(self):
        cfg = GVendiConfig()
        m = _make_manifest(cfg)
        with pytest.raises(ValueError):
            validate_selection_manifest(m, "random", 2, 2, 3, 3)

    def test_missing_cell_rejected(self):
        cfg = GVendiConfig()
        m = _make_manifest(cfg)
        m["selected_clients"] = m["selected_clients"][:-1]
        with pytest.raises(ValueError):
            validate_selection_manifest(m, "h_vendi", 2, 2, 3, 3)

    def test_duplicate_cell_rejected(self):
        cfg = GVendiConfig()
        m = _make_manifest(cfg)
        m["selected_clients"][-1] = dict(m["selected_clients"][-2])
        with pytest.raises(ValueError):
            validate_selection_manifest(m, "h_vendi", 2, 2, 3, 3)

    def test_try_load_roundtrip(self, tmp_path):
        cfg = GVendiConfig()
        m = _make_manifest(cfg)
        path = str(tmp_path / "manifest.json")
        atomic_write_json(path, m)
        loaded = try_load_selection_manifest(path, "h_vendi", 2, 2, 3, 3)
        assert loaded == m

    def test_try_load_missing_returns_none(self, tmp_path):
        assert try_load_selection_manifest(
            str(tmp_path / "nope.json"), "h_vendi", 2, 2, 3, 3
        ) is None

    def test_try_load_corrupt_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ not json")
        assert try_load_selection_manifest(
            str(path), "h_vendi", 2, 2, 3, 3
        ) is None

    def test_try_load_wrong_method_returns_none(self, tmp_path):
        cfg = GVendiConfig()
        m = _make_manifest(cfg, method="random")
        path = str(tmp_path / "manifest.json")
        atomic_write_json(path, m)
        assert try_load_selection_manifest(path, "h_vendi", 2, 2, 3, 3) is None


class TestSelectedClientRow:
    def test_serialization(self):
        spec = CandidateSpec(
            "cand_r00_003", 0, {"family": "trend_seasonal"}, "perturb", 905_000, [1]
        )
        ev = CandidateEvaluation(
            spec=spec,
            raw_feature_vector=torch.zeros(3),
            feature_vector=torch.tensor([0.1, 0.2, 0.3]),
            projected_gradients=torch.zeros(2, 4),
            proxy_metrics={"mase": 1.5},
            learnability_improvement=0.2,
            filter_results={},
            is_eligible=True,
        )
        sc = SelectionCandidate(
            "cand_r00_003", 0, torch.zeros(2, 4), ev.feature_vector, 1.5,
            sparse_fraction=0.5, centroid_distance=0.7,
            marginal_entropy_gain=0.01,
        )
        row = selected_client_row(ev, sc, 0, 7)
        assert row["client_key"] == "regime_00_variant_07"
        assert row["proxy_mase"] == 1.5
        assert row["marginal_entropy_gain"] == 0.01
        assert row["data_seed"] == 905_000
        json.dumps(row)  # must be JSON-serializable


class TestPaths:
    def test_layout(self, tmp_path):
        paths = GVendiPaths(str(tmp_path), 6).ensure()
        assert os.path.isdir(paths.stage_dir)
        assert paths.stage_dir.endswith("gvendi_stage_6")
        assert paths.selection_manifest("h_vendi").endswith(
            "selection_manifest_h_vendi.json"
        )
        assert paths.selection_manifest("random") != paths.selection_manifest("hard")
        assert paths.train_dir("h_vendi").endswith("train_h_vendi")
