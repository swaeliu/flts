"""Chained variant-only expansion test: 2x2 -> 2x3 -> 2x4 with a tiny model.

Exercises the variant-only selection mode (quota 1 per regime, no new
regime), the manifest cell layout, the cumulative-selection chaining used by
run_gvendi_variant_expansion (rebuilding the active set of a stage that was
itself manifest-built), boundary expansion, and one training round.
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
from gvendi.manifest import GVendiPaths, validate_selection_manifest
from gvendi.probing import ProbeEngine
from gvendi.run_gvendi_8x8 import run_selection
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
    cfg.clients_per_round = 4
    cfg.relora_lr_warmup_steps = 1

    cfg.gvendi_projection_dim = 16
    cfg.gvendi_probe_batches = 1
    cfg.gvendi_probe_batch_size = 8
    cfg.gvendi_candidates_per_existing_regime = 2
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


def _run_variant_selection(cfg, stage_dir, stage_index, server, engine,
                           projector, active, prev_v):
    active_clients, active_meta, active_features, feature_stats = active
    paths = GVendiPaths(stage_dir, stage_index).ensure()
    manifest = run_selection(
        cfg, paths, engine, projector, server,
        active_clients, active_meta, active_features, feature_stats,
        tiny_ctor, cfg._base_state, cfg._lora_cfg, SEQ_LEN,
        2, prev_v, 2, prev_v + 1,
        source_checkpoint="/fake/_relora_round_0_ckpt",
    )
    return paths, manifest


@pytest.fixture(scope="module")
def chained(tmp_path_factory):
    stage_dir = str(tmp_path_factory.mktemp("gvendi_vexp"))
    cfg = _make_cfg()
    validate_gvendi_config(cfg, 2, 2, 2, 3)

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

    # ---- stage 1: 2x2 -> 2x3 ------------------------------------------
    paths1, manifest1 = _run_variant_selection(
        cfg, os.path.join(stage_dir, "s1"), 1, server, engine, projector,
        active_22, prev_v=2,
    )
    cum_rows = list(manifest1["selected_clients"])

    # ---- chained active rebuild of the 2x3 stage ----------------------
    cfg2 = copy.deepcopy(cfg)
    cfg2.synthetic_num_regimes = 2
    cfg2.synthetic_variants_per_regime = 3
    cfg2.synthetic_num_clients = 6
    cfg2.n_clients = 6
    set_seed(cfg2.seed)
    active_23 = build_stage_clients_from_manifest(
        cfg2, cum_rows, 2, 2, 2, 3, seq_len=SEQ_LEN, batch_size=cfg2.batch_size,
    )
    server2 = _make_server(cfg2, active_23[2], flatdim)

    # ---- stage 2: 2x3 -> 2x4 ------------------------------------------
    paths2, manifest2 = _run_variant_selection(
        cfg2, os.path.join(stage_dir, "s2"), 2, server2, engine, projector,
        active_23, prev_v=3,
    )
    cum_rows2 = cum_rows + list(manifest2["selected_clients"])

    return {
        "cfg": cfg,
        "cfg2": cfg2,
        "spec": spec,
        "flatdim": flatdim,
        "base_state": base_state,
        "global_head_state": global_head_state,
        "server": server,
        "active_22": active_22,
        "active_23": active_23,
        "manifest1": manifest1,
        "manifest2": manifest2,
        "cum_rows": cum_rows,
        "cum_rows2": cum_rows2,
        "stage_dir": stage_dir,
        "paths1": paths1,
    }


class TestVariantOnlySelection:
    def test_manifest_shape_stage1(self, chained):
        rows = chained["manifest1"]["selected_clients"]
        assert len(rows) == 2  # one new variant per regime, no new regime
        cells = {(r["regime_index"], r["variant_index"]) for r in rows}
        assert cells == {(0, 2), (1, 2)}
        validate_selection_manifest(chained["manifest1"], "h_vendi", 2, 2, 2, 3)

    def test_manifest_shape_stage2(self, chained):
        rows = chained["manifest2"]["selected_clients"]
        assert len(rows) == 2
        cells = {(r["regime_index"], r["variant_index"]) for r in rows}
        assert cells == {(0, 3), (1, 3)}
        validate_selection_manifest(chained["manifest2"], "h_vendi", 2, 3, 2, 4)

    def test_candidate_pool_has_no_new_regime(self, chained):
        from gvendi.manifest import load_candidate_pool
        pool = load_candidate_pool(chained["paths1"].candidate_pool)
        assert all(int(c["regime_index"]) < 2 for c in pool)

    def test_chained_active_rebuild_shapes(self, chained):
        clients, meta_rows, features, _ = chained["active_23"]
        assert len(clients) == 6
        assert features.shape[0] == 6
        keys = {m["regime_variant"] for m in meta_rows}
        assert len(keys) == 6
        # the two manifest-selected clients occupy variant 2
        manifest_cells = [m for m in meta_rows if m["variant_id"] == 2]
        assert len(manifest_cells) == 2
        assert all(m["source_candidate_id"] for m in manifest_cells)

    def test_grid_clients_bit_identical_across_rebuilds(self, chained):
        """Grid cells of the 2x3 rebuild match the original 2x2 build."""
        _, meta22, feats22, _ = chained["active_22"]
        _, meta23, feats23, _ = chained["active_23"]
        idx22 = {m["regime_variant"]: i for i, m in enumerate(meta22)}
        idx23 = {m["regime_variant"]: i for i, m in enumerate(meta23)}
        for key, i22 in idx22.items():
            g22 = chained["active_22"][0][i22]["train"].dataset
            g23 = chained["active_23"][0][idx23[key]]["train"].dataset
            assert len(g22) == len(g23)
            x22, y22 = g22[0][0], g22[0][1]
            x23, y23 = g23[0][0], g23[0][1]
            assert torch.equal(x22, x23)
            assert torch.equal(y22, y23)


class TestVariantBoundaryAndTraining:
    @pytest.fixture(scope="class")
    def trained(self, chained):
        cfg = copy.deepcopy(chained["cfg"])
        server = chained["server"]

        clients, meta_rows, client_features, _ = (
            chained["active_23"][0],
            chained["active_23"][1],
            chained["active_23"][2],
            chained["active_23"][3],
        )

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
        cfg.synthetic_num_regimes = 2
        cfg.synthetic_variants_per_regime = 3

        new_client_ids = {
            cid for cid, m in enumerate(meta_rows) if int(m["variant_id"]) == 2
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
        assert trained["server"].emb.num_embeddings == 6
        assert trained["server"].client_features.shape[0] == 6

    def test_checkpoint_variant_tracking(self, chained):
        """track_checkpoint_variants=True returns all 5 selection-criterion
        variants without changing server's final restored state."""
        cfg = copy.deepcopy(chained["cfg"])
        clients, meta_rows, client_features, _ = chained["active_23"]

        server = _make_server(cfg, chained["active_22"][2], chained["flatdim"])
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
        cfg.synthetic_num_regimes = 2
        cfg.synthetic_variants_per_regime = 3
        new_client_ids = {
            cid for cid, m in enumerate(meta_rows) if int(m["variant_id"]) == 2
        }
        regime_labels = torch.tensor(
            [int(m["regime_id"]) for m in meta_rows], dtype=torch.long
        )
        scheduler = CurriculumScheduler(cfg)
        tracker = ClientPatienceTracker(patience=50)

        csv_path = os.path.join(chained["stage_dir"], "history_variants.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GVENDI_CSV_FIELDNAMES)
            writer.writeheader()
            final_metrics, head, last_row, variants = run_gvendi_stage(
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
                max_comm_rounds=3,
                new_client_ids=new_client_ids,
                track_checkpoint_variants=True,
            )

        assert variants is not None
        for key in (
            "global_best", "new_best", "cohort_balanced_best",
            "constrained_best", "final",
        ):
            assert variants[key] is not None, key
            assert "hnet" in variants[key] and "emb" in variants[key]
            assert "comm_round" in variants[key]
            assert variants[key]["metrics"] is not None
        assert "old_client_val_mae" in variants["cohort_balanced_best"]["metrics"]
        assert "new_client_val_mae" in variants["constrained_best"]["metrics"]

        # server's returned state (post-restore-best) must match global_best,
        # not final -- tracking must not change existing restore behavior.
        gb_head = variants["global_best"]["head"]
        assert set(gb_head.keys()) == set(head.keys())
        for k in gb_head:
            assert torch.equal(gb_head[k], head[k])

    def test_stage_trained(self, trained):
        fm = trained["final_metrics"]
        assert fm is not None
        assert fm["comm_round"] >= 1
        assert np.isfinite(fm["mean_client_mae"])
        assert len(trained["new_client_ids"]) == 2
        assert fm["new_client_mean_mae"] is not None
        assert fm["old_client_mean_mae"] is not None
