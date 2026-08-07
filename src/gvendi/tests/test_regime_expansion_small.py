"""Chained rectangular expansion test: 2x2 -> 3x3 -> 4x4 with a tiny model.

Exercises the rectangular selection mode (new regime + one new variant per
existing regime), and specifically the multi-hop cumulative reconstruction:
load_cumulative_selection/build_active_clients must correctly rebuild a
population that grew in BOTH regime and variant dimensions across hops,
using the TRUE base grid consistently rather than drifting per-hop (see
run_gvendi_regime_expansion.py's module docstring for why this is a real
correctness risk that variant-only chaining doesn't have).
"""

import copy
import csv
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from synthetic_data import make_synthetic_clients, set_seed
from server import Server
from lora_utils import (
    get_lora_spec_and_flatdim,
    inject_lora,
    mark_only_lora_trainable,
)

from relora.curriculum import (
    CurriculumScheduler,
    _rebuild_optimizer_with_new_emb,
    expand_server_embeddings,
)
from relora.patience import ClientPatienceTracker

from gvendi.config import GVendiConfig, validate_gvendi_config
from gvendi.data import build_stage_clients_from_manifest
from gvendi.grad_vendi import RademacherProjector
from gvendi.manifest import GVendiPaths, atomic_write_json, validate_selection_manifest
from gvendi.probing import ProbeEngine
from gvendi.run_gvendi_8x8 import run_selection
from gvendi.run_gvendi_variant_expansion import (
    build_active_clients,
    load_cumulative_selection,
)
from gvendi.train_8x8 import GVENDI_CSV_FIELDNAMES, run_gvendi_stage

SEQ_LEN = 64
HORIZON = 32
DEVICE = torch.device("cpu")


class TinyForecastModel(nn.Module):
    def __init__(self, seq_len: int = SEQ_LEN, horizon: int = HORIZON):
        super().__init__()
        self.config = SimpleNamespace(seq_len=seq_len)
        self.encoder = nn.Linear(seq_len, 24)
        self.forecast_head = nn.Linear(24, horizon)

    def forward(self, x_enc):
        h = torch.relu(self.encoder(x_enc.squeeze(1)))
        return SimpleNamespace(forecast=self.forecast_head(h).unsqueeze(1))


def tiny_ctor():
    torch.manual_seed(1234)
    model = TinyForecastModel()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _make_cfg():
    cfg = GVendiConfig()
    cfg.seed = 3
    cfg.out_dir = None
    cfg.synthetic_series_per_client = 2
    cfg.synthetic_gp_samples_per_client = 1
    cfg.synthetic_mixup_per_client = 1
    cfg.batch_size = 8
    cfg.eval_batches = 2
    cfg.local_steps = 1
    cfg.clients_per_round = 16
    cfg.relora_lr_warmup_steps = 1

    cfg.gvendi_projection_dim = 16
    cfg.gvendi_probe_batches = 1
    cfg.gvendi_probe_batch_size = 8
    cfg.gvendi_candidates_per_existing_regime = 2
    cfg.gvendi_candidates_new_regime = 6
    cfg.gvendi_num_clusters = 4
    cfg.gvendi_regime_order_trials = 2
    cfg.gvendi_learnability_steps = 1
    cfg.gvendi_min_learnability = -1000.0  # tiny model: don't gate on this
    cfg.gvendi_mase_low_quantile = 0.0
    cfg.gvendi_mase_high_quantile = 1.0
    cfg.gvendi_probe_series_per_client = 2
    cfg.gvendi_probe_gp_samples = 1
    cfg.gvendi_probe_mixup_per_client = 1
    return cfg


def _model_parts(cfg):
    model0 = tiny_ctor()
    inject_lora(model0, r=cfg.lora_rank, alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout, exclude_keywords=cfg.exclude_keywords)
    mark_only_lora_trainable(model0)
    spec, flatdim = get_lora_spec_and_flatdim(model0)
    base_state = tiny_ctor().state_dict()
    from client import extract_forecast_head_state_dict
    global_head_state = extract_forecast_head_state_dict(tiny_ctor())
    del model0
    lora_cfg = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "exclude_keywords": cfg.exclude_keywords,
    }
    return spec, flatdim, base_state, global_head_state, lora_cfg


def _make_server(cfg, features, flatdim):
    torch.manual_seed(cfg.seed)
    return Server(
        n_clients=features.shape[0],
        emb_dim=int(features.shape[1]),
        hidden=32,
        flat_dim=flatdim,
        lr=1e-3,
        device=DEVICE,
        client_features=features,
        hnet_dropout=0.0,
        learnable_embeddings=False,
    )


def _run_rectangular_selection(cfg, stage_dir, stage_index, server, engine,
                               projector, active, prev_r, prev_v):
    active_clients, active_meta, active_features, feature_stats = active
    paths = GVendiPaths(stage_dir, stage_index).ensure()
    manifest = run_selection(
        cfg, paths, engine, projector, server,
        active_clients, active_meta, active_features, feature_stats,
        tiny_ctor, cfg._base_state, cfg._lora_cfg, SEQ_LEN,
        prev_r, prev_v, prev_r + 1, prev_v + 1,
        source_checkpoint="/fake/_relora_round_0_ckpt",
    )
    return paths, manifest


@pytest.fixture(scope="module")
def chained(tmp_path_factory):
    stage_dir = str(tmp_path_factory.mktemp("gvendi_rexp"))
    cfg = _make_cfg()
    validate_gvendi_config(cfg, 2, 2, 3, 3)

    cfg.synthetic_num_regimes = 2
    cfg.synthetic_variants_per_regime = 2
    cfg.synthetic_num_clients = 4
    cfg.n_clients = 4

    spec, flatdim, base_state, global_head_state, lora_cfg = _model_parts(cfg)
    cfg._base_state = base_state
    cfg._lora_cfg = lora_cfg

    set_seed(cfg.seed)
    active_22 = make_synthetic_clients(
        cfg, seq_len=SEQ_LEN, batch_size=cfg.batch_size,
        return_feature_stats=True,
    )
    server = _make_server(cfg, active_22[2], flatdim)
    engine = ProbeEngine(
        tiny_ctor, base_state, spec, flatdim, global_head_state,
        lora_cfg, DEVICE, loss_type=cfg.train_loss,
    )
    projector = RademacherProjector(
        input_dim=flatdim, output_dim=cfg.gvendi_projection_dim,
        seed=cfg.gvendi_projection_seed,
    )

    # ---- stage 1: 2x2 -> 3x3 -------------------------------------------
    paths1, manifest1 = _run_rectangular_selection(
        cfg, os.path.join(stage_dir, "s1"), 1, server, engine, projector,
        active_22, prev_r=2, prev_v=2,
    )
    cum_rows = list(manifest1["selected_clients"])

    # ---- chained active rebuild of the 3x3 stage ------------------------
    # This is the crux of the test: base grid stays 2x2 forever, but the
    # population being rebuilt is 3x3 -- regime count grew, not just
    # variants, unlike the variant-only chain.
    cfg2 = copy.deepcopy(cfg)
    cfg2.synthetic_num_regimes = 3
    cfg2.synthetic_variants_per_regime = 3
    cfg2.synthetic_num_clients = 9
    cfg2.n_clients = 9
    base_r, base_v, cum_rows_loaded = load_cumulative_selection(
        os.path.join(stage_dir, "s1_train"), 2, 2,
    )
    # No cumulative file exists yet at a fresh dir -> falls back to (2, 2, []).
    assert (base_r, base_v, cum_rows_loaded) == (2, 2, [])

    active_33 = build_active_clients(
        cfg2, 2, 2, 3, 3, cum_rows, SEQ_LEN,
    )
    server2 = _make_server(cfg2, active_33[2], flatdim)

    # ---- stage 2: 3x3 -> 4x4 --------------------------------------------
    paths2, manifest2 = _run_rectangular_selection(
        cfg2, os.path.join(stage_dir, "s2"), 2, server2, engine, projector,
        active_33, prev_r=3, prev_v=3,
    )
    cum_rows2 = cum_rows + list(manifest2["selected_clients"])

    # ---- re-derive the 3x3 population a SECOND way: via
    # load_cumulative_selection reading a persisted cumulative file, exactly
    # as run_gvendi_regime_expansion.py does on a real chained hop -------
    cum_file_dir = os.path.join(stage_dir, "s1_persisted")
    os.makedirs(cum_file_dir, exist_ok=True)
    atomic_write_json(
        os.path.join(cum_file_dir, "cumulative_selected_clients.json"),
        {"base_n_regimes": 2, "base_n_variants": 2, "rows": cum_rows},
    )
    reloaded_base_r, reloaded_base_v, reloaded_rows = load_cumulative_selection(
        cum_file_dir, 3, 3,
    )

    return {
        "cfg": cfg,
        "cfg2": cfg2,
        "spec": spec,
        "flatdim": flatdim,
        "base_state": base_state,
        "global_head_state": global_head_state,
        "server": server,
        "active_22": active_22,
        "active_33": active_33,
        "manifest1": manifest1,
        "manifest2": manifest2,
        "cum_rows": cum_rows,
        "cum_rows2": cum_rows2,
        "reloaded_base_r": reloaded_base_r,
        "reloaded_base_v": reloaded_base_v,
        "reloaded_rows": reloaded_rows,
        "stage_dir": stage_dir,
        "paths1": paths1,
    }


class TestRectangularSelection:
    def test_manifest_shape_stage1(self, chained):
        rows = chained["manifest1"]["selected_clients"]
        # 2 existing-regime new-variant picks + 3 new-regime variants = 5
        assert len(rows) == 5
        cells = {(r["regime_index"], r["variant_index"]) for r in rows}
        assert cells == {(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}
        validate_selection_manifest(chained["manifest1"], "h_vendi", 2, 2, 3, 3)

    def test_manifest_shape_stage2(self, chained):
        rows = chained["manifest2"]["selected_clients"]
        # 3 existing-regime new-variant picks + 4 new-regime variants = 7
        assert len(rows) == 7
        cells = {(r["regime_index"], r["variant_index"]) for r in rows}
        assert cells == {
            (0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3),
        }
        validate_selection_manifest(chained["manifest2"], "h_vendi", 3, 3, 4, 4)

    def test_chained_active_rebuild_shapes(self, chained):
        clients, meta_rows, features, _ = chained["active_33"]
        assert len(clients) == 9
        assert features.shape[0] == 9
        keys = {m["regime_variant"] for m in meta_rows}
        assert len(keys) == 9
        new_cells = [m for m in meta_rows if m["is_new_client"]]
        assert len(new_cells) == 5  # matches stage1's manifest row count
        assert all(m["source_candidate_id"] for m in new_cells)

    def test_original_grid_cells_bit_identical_across_rebuilds(self, chained):
        """The 2x2 original cells must use the TRUE base (2x2) params in the
        3x3 rebuild, not get regenerated as if prev_n_regimes were 3 -- that
        drift is exactly the correctness risk this test guards against."""
        _, meta22, feats22, _ = chained["active_22"]
        _, meta33, feats33, _ = chained["active_33"]
        idx22 = {m["regime_variant"]: i for i, m in enumerate(meta22)}
        idx33 = {m["regime_variant"]: i for i, m in enumerate(meta33)}
        for key, i22 in idx22.items():
            g22 = chained["active_22"][0][i22]["train"].dataset
            g33 = chained["active_33"][0][idx33[key]]["train"].dataset
            assert len(g22) == len(g33)
            x22, y22 = g22[0][0], g22[0][1]
            x33, y33 = g33[0][0], g33[0][1]
            assert torch.equal(x22, x33)
            assert torch.equal(y22, y33)

    def test_cumulative_file_reload_matches_in_memory_chain(self, chained):
        """load_cumulative_selection reading a persisted file must recover
        exactly the base grid + rows an in-memory chain already has."""
        assert chained["reloaded_base_r"] == 2
        assert chained["reloaded_base_v"] == 2
        assert chained["reloaded_rows"] == chained["cum_rows"]

    def test_second_hop_new_client_ids_exclude_first_hop_clients(self, chained):
        """Regression test for a real bug: on a hop past the first, 'new at
        THIS boundary' must be computed against prev_r/prev_v (the immediate
        predecessor shape), not meta_rows' 'is_new_client' flag -- that flag
        is relative to the constant TRUE base grid (correct for bit-identical
        reconstruction, Section test above) and would incorrectly re-count
        every earlier hop's already-trained clients as new again on hop 2+.
        Caught live: run_gvendi_regime_expansion.py originally used
        'is_new_client' directly and reported 50 new / 14 carried over for a
        7x7->8x8 hop that should have been 15 new / 49 carried over.
        """
        cfg2 = chained["cfg2"]
        clients44, meta_rows44, features44, _ = build_stage_clients_from_manifest(
            cfg2, chained["cum_rows2"], 2, 2, 4, 4,
            seq_len=SEQ_LEN, batch_size=cfg2.batch_size,
        )
        assert len(clients44) == 16

        # Correct: new relative to hop 2's predecessor (3x3) -- only the 7
        # cells hop 2's own manifest selected (3 existing-regime picks + 4
        # new-regime variants), matching manifest2's row count exactly.
        prev_r, prev_v = 3, 3
        correct_new = {
            cid for cid, m in enumerate(meta_rows44)
            if int(m["regime_id"]) >= prev_r or int(m["variant_id"]) >= prev_v
        }
        assert len(correct_new) == len(chained["manifest2"]["selected_clients"]) == 7

        # The bug: 'is_new_client' is relative to the true base (2x2), so it
        # also (wrongly) counts hop 1's 5 selected clients as "new" here.
        buggy_new = {
            cid for cid, m in enumerate(meta_rows44) if m["is_new_client"]
        }
        assert len(buggy_new) == 12
        assert buggy_new != correct_new


class TestRectangularBoundaryAndTraining:
    @pytest.fixture(scope="class")
    def trained(self, chained):
        cfg = copy.deepcopy(chained["cfg"])
        server = chained["server"]

        clients, meta_rows, client_features, _ = chained["active_33"]

        old_emb_param = server.emb.weight
        for group in server.opt.param_groups:
            group["lr"] = 0.0
        expand_server_embeddings(
            server, chained["active_22"][1], meta_rows,
            old_n_regimes=2, old_n_variants=2,
        )
        _rebuild_optimizer_with_new_emb(server, old_emb_param)
        server.client_features = client_features.to(DEVICE)

        cfg.n_clients = len(clients)
        cfg.synthetic_num_regimes = 3
        cfg.synthetic_variants_per_regime = 3

        new_client_ids = {
            cid for cid, m in enumerate(meta_rows) if m["is_new_client"]
        }
        regime_labels = torch.tensor(
            [int(m["regime_id"]) for m in meta_rows], dtype=torch.long
        )
        scheduler = CurriculumScheduler(cfg)
        tracker = ClientPatienceTracker(patience=50)

        csv_path = os.path.join(chained["stage_dir"], "history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GVENDI_CSV_FIELDNAMES)
            writer.writeheader()
            final_metrics, head, last_row, _ = run_gvendi_stage(
                curriculum_round=1,
                cfg=cfg,
                server=server,
                clients=clients,
                meta_rows=meta_rows,
                base_ctor=tiny_ctor,
                base_state=chained["base_state"],
                spec=chained["spec"],
                global_head_state=chained["global_head_state"],
                device=DEVICE,
                regime_labels=regime_labels,
                patience_tracker=tracker,
                scheduler=scheduler,
                warmup_step_counter=[0],
                target_lrs=[1e-3] * len(server.opt.param_groups),
                csv_writer=writer,
                csv_file=f,
                max_comm_rounds=2,
                new_client_ids=new_client_ids,
            )
        return {
            "final_metrics": final_metrics,
            "server": server,
            "new_client_ids": new_client_ids,
            "clients": clients,
        }

    def test_boundary_expanded_embeddings(self, trained):
        assert trained["server"].emb.num_embeddings == 9
        assert trained["server"].client_features.shape[0] == 9

    def test_stage_trained(self, trained):
        fm = trained["final_metrics"]
        assert fm is not None
        assert fm["comm_round"] >= 1
        assert np.isfinite(fm["mean_client_mae"])
        assert len(trained["new_client_ids"]) == 5
        assert fm["new_client_mean_mae"] is not None
        assert fm["old_client_mean_mae"] is not None
