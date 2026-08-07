"""End-to-end 2x2 -> 3x3 integration test with a tiny stand-in model.

Exercises the complete augmentation path — post-restore probing, candidate
pool, quality gate, shortlisting, selection (all three methods), manifest
persistence + resume, manifest-driven client construction, boundary
expansion, and at least one training communication round — without the MOMENT
backbone.
"""

import copy
import csv
import io
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
from gvendi.manifest import (
    GVendiPaths,
    load_candidate_pool,
    try_load_selection_manifest,
    validate_selection_manifest,
)
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
    cfg.synthetic_num_regimes = 2
    cfg.synthetic_variants_per_regime = 2
    cfg.synthetic_num_clients = 4
    cfg.n_clients = 4
    cfg.synthetic_series_per_client = 2
    cfg.synthetic_gp_samples_per_client = 1
    cfg.synthetic_mixup_per_client = 1
    cfg.batch_size = 8
    cfg.eval_batches = 2
    cfg.local_steps = 1
    cfg.clients_per_round = 4
    cfg.relora_lr_warmup_steps = 1

    # gvendi: four active clients, two candidates per existing regime, six
    # for the new regime, small projection, one probe batch, one local step.
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


@pytest.fixture(scope="module")
def stage_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("gvendi_stage"))


@pytest.fixture(scope="module")
def pipeline(stage_dir):
    cfg = _make_cfg()
    validate_gvendi_config(cfg, 2, 2, 3, 3)

    set_seed(cfg.seed)
    active_clients, active_meta, active_features, feature_stats = (
        make_synthetic_clients(
            cfg, seq_len=SEQ_LEN, batch_size=cfg.batch_size,
            return_feature_stats=True,
        )
    )
    assert len(active_clients) == 4

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

    torch.manual_seed(cfg.seed)
    server = Server(
        n_clients=4,
        emb_dim=int(active_features.shape[1]),
        hidden=32,
        flat_dim=flatdim,
        lr=1e-3,
        device=DEVICE,
        client_features=active_features,
        hnet_dropout=0.0,
        learnable_embeddings=False,
    )

    engine = ProbeEngine(
        tiny_ctor, base_state, spec, flatdim, global_head_state,
        lora_cfg, DEVICE, loss_type=cfg.train_loss,
    )
    projector = RademacherProjector(
        input_dim=flatdim, output_dim=cfg.gvendi_projection_dim,
        seed=cfg.gvendi_projection_seed,
    )

    paths = GVendiPaths(stage_dir, 1).ensure()
    manifest = run_selection(
        cfg, paths, engine, projector, server,
        active_clients, active_meta, active_features, feature_stats,
        tiny_ctor, base_state, lora_cfg, SEQ_LEN,
        2, 2, 3, 3,
        source_checkpoint="/fake/_relora_round_0_ckpt",
    )
    return {
        "cfg": cfg,
        "paths": paths,
        "manifest": manifest,
        "server": server,
        "engine": engine,
        "projector": projector,
        "active_clients": active_clients,
        "active_meta": active_meta,
        "active_features": active_features,
        "feature_stats": feature_stats,
        "spec": spec,
        "flatdim": flatdim,
        "base_state": base_state,
        "global_head_state": global_head_state,
        "lora_cfg": lora_cfg,
    }


class TestSelectionPipeline:
    def test_restored_model_was_probed(self, pipeline):
        paths = pipeline["paths"]
        assert os.path.exists(paths.active_gradients)
        payload = torch.load(paths.active_gradients, weights_only=False)
        grads = payload["gradients"]
        assert len(grads) == 4
        for g in grads.values():
            assert g.shape == (1, 16)
            assert torch.allclose(g.norm(dim=1), torch.ones(1), atol=1e-4)

    def test_probing_did_not_modify_model(self, pipeline):
        # The probe engine's backbone weights must equal the base state.
        engine = pipeline["engine"]
        base_state = pipeline["base_state"]
        model_sd = engine.model.state_dict()
        for k, v in base_state.items():
            assert torch.equal(model_sd[k], v), k

    def test_correct_number_selected(self, pipeline):
        rows = pipeline["manifest"]["selected_clients"]
        assert len(rows) == 5  # 2 existing regimes + 3 new-regime variants
        validate_selection_manifest(pipeline["manifest"], "h_vendi", 2, 2, 3, 3)

    def test_manifest_persisted_and_resumable(self, pipeline):
        paths = pipeline["paths"]
        loaded = try_load_selection_manifest(
            paths.selection_manifest("h_vendi"), "h_vendi", 2, 2, 3, 3
        )
        assert loaded == pipeline["manifest"]

    def test_selection_artifacts_written(self, pipeline):
        paths = pipeline["paths"]
        for p in (paths.candidate_pool, paths.filter_report,
                  paths.kmeans_state, paths.candidate_evals,
                  paths.selection_report("h_vendi")):
            assert os.path.exists(p), p

    def test_selection_report_contents(self, pipeline):
        import json
        with open(pipeline["paths"].selection_report("h_vendi")) as f:
            report = json.load(f)
        assert report["active_h_vendi"] >= 1.0
        assert report["post_selection_h_vendi"] >= 1.0
        assert len(report["cluster_occupancy"]) == 4
        assert report["total_selection_wall_clock_secs"] > 0

    def test_selection_is_deterministic_on_rerun(self, pipeline):
        """With caches intact and the manifest removed, re-selection is identical."""
        paths = pipeline["paths"]
        cfg = pipeline["cfg"]
        manifest_path = paths.selection_manifest("h_vendi")
        original_ids = [
            r["candidate_id"] for r in pipeline["manifest"]["selected_clients"]
        ]
        os.remove(manifest_path)
        manifest2 = run_selection(
            cfg, paths, pipeline["engine"], pipeline["projector"],
            pipeline["server"], pipeline["active_clients"],
            pipeline["active_meta"], pipeline["active_features"],
            pipeline["feature_stats"], tiny_ctor, pipeline["base_state"],
            pipeline["lora_cfg"], SEQ_LEN, 2, 2, 3, 3,
            source_checkpoint="/fake/_relora_round_0_ckpt",
        )
        assert [r["candidate_id"] for r in manifest2["selected_clients"]] == \
            original_ids

    def test_random_and_hard_share_the_candidate_pool(self, pipeline):
        paths = pipeline["paths"]
        pool_before = load_candidate_pool(paths.candidate_pool)
        for method in ("random", "hard"):
            cfg = copy.deepcopy(pipeline["cfg"])
            cfg.gvendi_selection_method = method
            manifest = run_selection(
                cfg, paths, pipeline["engine"], pipeline["projector"],
                pipeline["server"], pipeline["active_clients"],
                pipeline["active_meta"], pipeline["active_features"],
                pipeline["feature_stats"], tiny_ctor, pipeline["base_state"],
                pipeline["lora_cfg"], SEQ_LEN, 2, 2, 3, 3,
                source_checkpoint="/fake/_relora_round_0_ckpt",
            )
            validate_selection_manifest(manifest, method, 2, 2, 3, 3)
            assert len(manifest["selected_clients"]) == 5
        assert load_candidate_pool(paths.candidate_pool) == pool_before


class TestBoundaryAndTraining:
    @pytest.fixture(scope="class")
    def trained(self, pipeline, stage_dir):
        cfg = copy.deepcopy(pipeline["cfg"])
        server = pipeline["server"]

        clients, meta_rows, client_features, _ = build_stage_clients_from_manifest(
            cfg, pipeline["manifest"]["selected_clients"],
            2, 2, 3, 3, seq_len=SEQ_LEN, batch_size=cfg.batch_size,
        )

        old_emb_param = server.emb.weight
        for group in server.opt.param_groups:
            group["lr"] = 0.0
        expand_server_embeddings(
            server, pipeline["active_meta"], meta_rows,
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

        csv_path = os.path.join(stage_dir, "history.csv")
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
                base_state=pipeline["base_state"],
                spec=pipeline["spec"],
                global_head_state=pipeline["global_head_state"],
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
            "cfg": cfg,
            "server": server,
            "clients": clients,
            "meta_rows": meta_rows,
            "client_features": client_features,
            "new_client_ids": new_client_ids,
            "final_metrics": final_metrics,
            "csv_path": csv_path,
        }

    def test_boundary_expanded_embeddings_and_features(self, trained):
        server = trained["server"]
        assert server.emb.num_embeddings == 9
        assert server.client_features.shape == (9, 17)

    def test_stage_trained_at_least_one_round(self, trained):
        fm = trained["final_metrics"]
        assert fm is not None
        assert fm["comm_round"] >= 1
        assert np.isfinite(fm["mean_client_mae"])

    def test_old_new_split_metrics_recorded(self, trained):
        fm = trained["final_metrics"]
        assert fm["old_client_mean_mae"] is not None
        assert fm["new_client_mean_mae"] is not None
        assert fm["old_client_mean_mase"] is not None
        assert fm["new_client_mean_mase"] is not None
        assert fm["worst_regime_mase"] is not None
        assert fm["new_client_sampling_multiplier"] >= 1.0

    def test_csv_has_gvendi_columns(self, trained):
        with open(trained["csv_path"]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1
        assert "old_client_mean_mae" in rows[0]
        assert "new_client_sampling_multiplier" in rows[0]
        assert float(rows[0]["new_client_sampling_multiplier"]) >= 1.0

    def test_exactly_nine_clients(self, trained):
        assert len(trained["clients"]) == 9
        assert len(trained["new_client_ids"]) == 5
