"""Tests that manifest-driven generation reproduces the legacy generator."""

import numpy as np
import pytest
import torch

from gvendi.config import GVendiConfig
from gvendi.data import (
    build_one_client_series,
    build_stage_clients_from_manifest,
    candidate_data_seed,
    compute_total_len,
    make_grid_params,
    make_new_regime_params,
)

from synthetic_data import build_synthetic_client_series


def _small_cfg():
    cfg = GVendiConfig()
    cfg.seed = 3
    cfg.synthetic_num_regimes = 2
    cfg.synthetic_variants_per_regime = 2
    cfg.synthetic_num_clients = 4
    cfg.synthetic_series_per_client = 2
    cfg.synthetic_gp_samples_per_client = 1
    cfg.synthetic_mixup_per_client = 1
    cfg.eval_batches = 2
    return cfg


SEQ_LEN = 64


class TestLegacyReproduction:
    def test_single_client_series_bit_identical(self):
        """build_one_client_series with legacy seeds == legacy generator output."""
        cfg = _small_cfg()
        legacy_clients, _ = build_synthetic_client_series(cfg, seq_len=SEQ_LEN)
        total_len = compute_total_len(cfg, SEQ_LEN)

        for client_id, (regime_id, variant_id) in enumerate(
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        ):
            params = make_grid_params(cfg, regime_id, variant_id, 2)
            series, ks_feat = build_one_client_series(
                cfg,
                params,
                rng_seed=cfg.seed + 1000 + client_id,
                series_seed_base=cfg.seed + client_id * 100,
                total_len=total_len,
                n_series=cfg.synthetic_series_per_client,
                n_gp=cfg.synthetic_gp_samples_per_client,
                n_mixup=cfg.synthetic_mixup_per_client,
            )
            legacy_series = legacy_clients[client_id]["series"]
            assert len(series) == len(legacy_series)
            for mine, theirs in zip(series, legacy_series):
                np.testing.assert_array_equal(mine, theirs)
            for k, v in ks_feat.items():
                assert legacy_clients[client_id]["meta"][k] == pytest.approx(v)

    def test_total_len_matches_legacy_minimum(self):
        cfg = _small_cfg()
        cfg.synthetic_series_length = 10  # far below the minimum
        total = compute_total_len(cfg, SEQ_LEN)
        # Legacy: max(seq+h+margin, 2*(seq+h), 4*(seq+h))
        assert total == 4 * (SEQ_LEN + cfg.horizon)


class TestNewRegimeParams:
    def test_default_family_follows_cycle(self):
        cfg = _small_cfg()
        params = make_new_regime_params(cfg, 7, 0, 8)
        assert params["family"] == "trend_seasonal"  # 7 % 7 == 0

    def test_family_override(self):
        cfg = _small_cfg()
        params = make_new_regime_params(
            cfg, 7, 0, 8, family_override="high_frequency"
        )
        assert params["family"] == "high_frequency"

    def test_unknown_family_raises(self):
        cfg = _small_cfg()
        with pytest.raises(ValueError):
            make_new_regime_params(cfg, 7, 0, 8, family_override="nope")

    def test_candidate_index_varies_params(self):
        cfg = _small_cfg()
        a = make_new_regime_params(cfg, 2, 0, 3)
        b = make_new_regime_params(cfg, 2, 1, 3)
        assert a != b


def _fake_manifest_rows(cfg, prev_r=2, prev_v=2, new_v=3):
    """Selected clients for a 2x2 -> 3x3 expansion, from real generator params."""
    rows = []
    idx = 0
    for r in range(prev_r):
        rows.append({
            "candidate_id": f"cand_r{r:02d}_000",
            "regime_index": r,
            "variant_index": prev_v,
            "generator_parameters": make_grid_params(cfg, r, 50 + r, prev_r),
            "data_seed": candidate_data_seed(cfg.seed, idx),
        })
        idx += 1
    for v in range(new_v):
        rows.append({
            "candidate_id": f"cand_r{prev_r:02d}_{v:03d}",
            "regime_index": prev_r,
            "variant_index": v,
            "generator_parameters": make_new_regime_params(cfg, prev_r, v, prev_r + 1),
            "data_seed": candidate_data_seed(cfg.seed, idx),
        })
        idx += 1
    return rows


class TestStageBuild:
    def test_grid_shape_ordering_and_flags(self):
        cfg = _small_cfg()
        rows = _fake_manifest_rows(cfg)
        clients, meta, feats, stats = build_stage_clients_from_manifest(
            cfg, rows, 2, 2, 3, 3, seq_len=SEQ_LEN, batch_size=8
        )
        assert len(clients) == 9
        assert len(meta) == 9
        assert feats.shape == (9, 17)
        assert [m["client_id"] for m in meta] == list(range(9))
        for m in meta:
            expected_new = m["regime_id"] == 2 or m["variant_id"] == 2
            assert m["is_new_client"] == expected_new
            if expected_new:
                assert m["source_candidate_id"] is not None
            else:
                assert m["source_candidate_id"] is None
        assert sum(m["is_new_client"] for m in meta) == 5

    def test_features_normalized_and_stats_returned(self):
        cfg = _small_cfg()
        rows = _fake_manifest_rows(cfg)
        _, _, feats, stats = build_stage_clients_from_manifest(
            cfg, rows, 2, 2, 3, 3, seq_len=SEQ_LEN, batch_size=8
        )
        assert torch.allclose(feats.mean(dim=0), torch.zeros(17), atol=1e-4)
        assert stats["mean"].shape == (17,)
        assert stats["std"].shape == (17,)
        assert torch.isfinite(feats).all()

    def test_reconstruction_is_deterministic(self):
        """The same manifest reconstructs the exact same clients."""
        cfg = _small_cfg()
        rows = _fake_manifest_rows(cfg)
        _, meta1, feats1, _ = build_stage_clients_from_manifest(
            cfg, rows, 2, 2, 3, 3, seq_len=SEQ_LEN, batch_size=8
        )
        _, meta2, feats2, _ = build_stage_clients_from_manifest(
            cfg, rows, 2, 2, 3, 3, seq_len=SEQ_LEN, batch_size=8
        )
        assert torch.equal(feats1, feats2)
        assert meta1 == meta2

    def test_originals_keep_previous_stage_data(self):
        """Original cells reuse their previous-stage series, not 3x3 regeneration."""
        cfg = _small_cfg()
        legacy_clients, _ = build_synthetic_client_series(cfg, seq_len=SEQ_LEN)
        rows = _fake_manifest_rows(cfg)
        clients, meta, _, _ = build_stage_clients_from_manifest(
            cfg, rows, 2, 2, 3, 3, seq_len=SEQ_LEN, batch_size=8
        )
        # Client (regime 1, variant 0): position 3 in the 3x3 grid, legacy id 2.
        new_pos = next(
            i for i, m in enumerate(meta)
            if m["regime_id"] == 1 and m["variant_id"] == 0
        )
        legacy_meta = legacy_clients[2]["meta"]
        assert meta[new_pos]["family"] == legacy_meta["family"]
        assert meta[new_pos]["trend_slope"] == pytest.approx(
            legacy_meta["trend_slope"]
        )
        assert meta[new_pos]["n_raw_series"] == legacy_meta["n_raw_series"]

    def test_missing_cell_raises(self):
        cfg = _small_cfg()
        rows = _fake_manifest_rows(cfg)[:-1]
        with pytest.raises(ValueError, match="no client for grid cell"):
            build_stage_clients_from_manifest(
                cfg, rows, 2, 2, 3, 3, seq_len=SEQ_LEN, batch_size=8
            )
