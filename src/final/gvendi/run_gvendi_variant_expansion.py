"""Chained variant-only H-Vendi expansion: R x V -> R x (V+1), regimes fixed.

One invocation performs one expansion stage:

    load the source stage (a plain ReLoRA run for the first expansion, or a
        previous variant-expansion stage), restore its checkpoint
    rebuild the source stage's client set bit-identically
        (base grid cells + every previously selected candidate, from the
        cumulative_selected_clients.json chain)
    run the shared gvendi selection pipeline in variant-only mode
        (1 new variant per regime, no new regime)
    execute the ordinary ReLoRA boundary and train the expanded stage
    persist the extended cumulative selection so the next stage can chain

Usage (7x2 -> 7x3; chain by pointing --source_dir at the previous stage's
train dir):

    python -m gvendi.run_gvendi_variant_expansion \
        --source_dir <relora_7x2_run_or_previous_stage_train_dir> \
        --out_dir <new_stage_dir> \
        --prev_round 0 \
        --n_regimes 7 --prev_n_variants 2 \
        --gvendi_selection_method h_vendi
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
for _sub in ("hypernet", "data_generation"):
    _p = os.path.join(_SRC_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from server import Server  # noqa: E402
from synthetic_data import make_synthetic_clients, set_seed  # noqa: E402

from curriculum.curriculum import (  # noqa: E402
    CurriculumScheduler,
    _rebuild_optimizer_with_new_emb,
    expand_server_embeddings,
)
from curriculum.patience import ClientPatienceTracker  # noqa: E402

from .config import (  # noqa: E402
    GVENDI_OVERRIDE_NAMES,
    GVendiConfig,
    add_gvendi_cli_args,
    validate_gvendi_config,
)
from . import dp  # noqa: E402
from .data import build_stage_clients_from_manifest  # noqa: E402
from .grad_vendi import RademacherProjector  # noqa: E402
from .manifest import (  # noqa: E402
    GVendiPaths,
    atomic_write_json,
    load_json,
    try_load_selection_manifest,
)
from .probing import ProbeEngine  # noqa: E402
from .run_gvendi_8x8 import _build_base_model, _dev, run_selection  # noqa: E402
from .train_8x8 import GVENDI_CSV_FIELDNAMES, run_gvendi_stage  # noqa: E402

CUMULATIVE_FILENAME = "cumulative_selected_clients.json"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Variant-only H-Vendi client expansion (R x V -> R x V+1)"
    )
    p.add_argument("--source_dir", type=str, required=True,
                   help="Directory of the source stage: a ReLoRA out_dir for "
                        "the first expansion, or the previous expansion "
                        "stage's train dir")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for this expansion stage")
    p.add_argument("--prev_round", type=int, required=True,
                   help="Curriculum-round index of the source stage "
                        "(source_dir/round_{prev_round}_config.json)")
    p.add_argument("--n_regimes", type=int, required=True)
    p.add_argument("--prev_n_variants", type=int, required=True)
    p.add_argument("--relora_client_patience", type=int, default=15)
    p.add_argument("--relora_max_rounds_per_curriculum", type=int, default=200)
    p.add_argument("--clients_per_round", type=int, default=None,
                   help="Override the inherited clients_per_round (clamped to "
                        "the stage's client count; pass a large value for "
                        "full participation). Default: inherit from the "
                        "source stage's config.")
    p.add_argument("--selection_only", action="store_true",
                   help="Write the selection manifest + report and stop")
    add_gvendi_cli_args(p)
    return p.parse_args()


def _load_cfg(source_dir: str, prev_round: int) -> GVendiConfig:
    path = os.path.join(source_dir, f"round_{prev_round}_config.json")
    with open(path) as f:
        d = json.load(f)
    for admin_key in ("curriculum_round", "n_regimes", "n_variants",
                      "n_clients", "target_lr", "selection_method"):
        d.pop(admin_key, None)
    cfg = GVendiConfig()
    for k, v in d.items():
        setattr(cfg, k, v)
    return cfg


def load_cumulative_selection(source_dir: str, prev_n_regimes: int,
                              prev_n_variants: int) -> tuple[int, int, list]:
    """Return (base_n_regimes, base_n_variants, cumulative selected rows).

    Absent file means the source stage IS the base grid (first expansion from
    a plain ReLoRA run). *prev_n_regimes*/*prev_n_variants* describe the
    ALREADY-BUILT population at source_dir -- its current grid shape, which
    may exceed the true base grid in the variant direction (variant-only
    chains), the regime direction, or both (rectangular chains); the base
    grid itself is read from the cumulative file, not assumed constant. The
    row count is validated generically as the cell-count difference between
    the current and base rectangles, which holds regardless of how the grid
    shape evolved hop to hop.
    """
    path = os.path.join(source_dir, CUMULATIVE_FILENAME)
    if not os.path.exists(path):
        return prev_n_regimes, prev_n_variants, []
    payload = load_json(path)
    base_r = int(payload["base_n_regimes"])
    base_v = int(payload["base_n_variants"])
    rows = payload["rows"]
    expected_rows = prev_n_regimes * prev_n_variants - base_r * base_v
    if len(rows) != expected_rows:
        raise ValueError(
            f"cumulative selection has {len(rows)} rows, expected "
            f"{expected_rows} for base {base_r}x{base_v} -> "
            f"current {prev_n_regimes}x{prev_n_variants}"
        )
    return base_r, base_v, rows


def _find_checkpoint_dir(source_dir: str, prev_round: int) -> str:
    for name in (f"_relora_round_{prev_round}_ckpt", "_relora_final_ckpt"):
        d = os.path.join(source_dir, name)
        if os.path.exists(os.path.join(d, "server.pt")):
            return d
    raise FileNotFoundError(
        f"no checkpoint found in {source_dir} "
        f"(_relora_round_{prev_round}_ckpt or _relora_final_ckpt)"
    )


def build_active_clients(cfg, base_r, base_v, n_regimes, prev_n_variants,
                         cum_rows, seq_len):
    """Rebuild the source stage's client set bit-identically."""
    cfg.synthetic_num_regimes = n_regimes
    cfg.synthetic_variants_per_regime = prev_n_variants
    cfg.synthetic_num_clients = n_regimes * prev_n_variants
    cfg.n_clients = cfg.synthetic_num_clients
    cfg.data_source = "synthetic"

    set_seed(cfg.seed)
    if not cum_rows:
        return make_synthetic_clients(
            cfg, seq_len=seq_len, batch_size=cfg.batch_size,
            return_feature_stats=True,
        )
    return build_stage_clients_from_manifest(
        cfg, cum_rows, base_r, base_v, n_regimes, prev_n_variants,
        seq_len=seq_len, batch_size=cfg.batch_size,
    )


def main():
    args = _parse_args()
    n_regimes = args.n_regimes
    prev_v = args.prev_n_variants
    new_v = prev_v + 1
    new_round = args.prev_round + 1

    cfg = _load_cfg(args.source_dir, args.prev_round)
    method_args = {
        name: getattr(args, name)
        for name in GVENDI_OVERRIDE_NAMES
        if getattr(args, name, None) is not None
    }
    for name, val in method_args.items():
        setattr(cfg, name, val)

    cfg.relora_client_patience = args.relora_client_patience
    cfg.relora_max_rounds_per_curriculum = args.relora_max_rounds_per_curriculum
    if args.clients_per_round is not None:
        cfg.clients_per_round = args.clients_per_round
    cfg.relora_max_rounds_schedule = ""
    cfg.relora_curriculum_schedule = (
        cfg.relora_curriculum_schedule.rstrip(",")
        + f",{n_regimes}x{new_v}"
    )

    validate_gvendi_config(cfg, n_regimes, prev_v, n_regimes, new_v)
    method = cfg.gvendi_selection_method

    device = _dev()
    print(f"[gvendi-vexp] device={device} method={method}")
    print(f"[gvendi-vexp] {n_regimes}x{prev_v} -> {n_regimes}x{new_v}")

    if not getattr(cfg, "use_fixed_client_features", False):
        raise NotImplementedError(
            "gvendi requires use_fixed_client_features=True (candidate "
            "adapters are generated from feature vectors)"
        )

    base_ctor, base_state, spec, flatdim, seq_len = _build_base_model(cfg, device)
    print(f"[gvendi-vexp] LoRA spec: {len(spec)} tensors, flatdim={flatdim:,}")
    lora_cfg = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "exclude_keywords": cfg.exclude_keywords,
    }

    # ------------------------------------------------------------------
    # Rebuild the active (source-stage) client set, deterministically.
    # ------------------------------------------------------------------
    base_r, base_v, cum_rows = load_cumulative_selection(
        args.source_dir, n_regimes, prev_v
    )
    active_clients, active_meta, active_features, feature_stats = (
        build_active_clients(
            cfg, base_r, base_v, n_regimes, prev_v, cum_rows, seq_len
        )
    )
    print(f"[gvendi-vexp] rebuilt {len(active_clients)} active clients "
          f"(base grid {base_r}x{base_v}, {len(cum_rows)} from manifests)")
    feat_dim = int(active_features.shape[1])
    if cfg.emb_dim != feat_dim:
        cfg.emb_dim = feat_dim

    # ------------------------------------------------------------------
    # Restore the source stage's checkpoint and freeze it for selection.
    # ------------------------------------------------------------------
    prev_ckpt_dir = _find_checkpoint_dir(args.source_dir, args.prev_round)
    server = Server(
        n_clients=cfg.n_clients,
        emb_dim=cfg.emb_dim,
        hidden=cfg.hnet_hidden,
        flat_dim=flatdim,
        lr=0.0,
        device=device,
        client_features=active_features,
        hnet_dropout=cfg.hnet_dropout,
        learnable_embeddings=bool(cfg.force_learnable_client_embeddings),
    )
    payload = torch.load(
        os.path.join(prev_ckpt_dir, "server.pt"), map_location=device,
        weights_only=False,
    )
    server.hnet.load_state_dict(payload["hnet"])
    server.emb.load_state_dict(payload["emb"])
    global_head_state = torch.load(
        os.path.join(prev_ckpt_dir, "global_head_state.pt"), map_location="cpu",
        weights_only=False,
    )
    print(f"[gvendi-vexp] restored stage-best state from {prev_ckpt_dir}")

    # ------------------------------------------------------------------
    # Selection (resume-safe): skip entirely when a valid manifest exists.
    # ------------------------------------------------------------------
    paths = GVendiPaths(args.out_dir, new_round).ensure()
    manifest = None
    if cfg.gvendi_resume_from_manifest:
        manifest = try_load_selection_manifest(
            paths.selection_manifest(method), method,
            n_regimes, prev_v, n_regimes, new_v,
        )
        if manifest is not None:
            print("[gvendi-vexp] valid selection manifest found; "
                  "skipping selection")

    if manifest is None:
        engine = ProbeEngine(
            base_ctor, base_state, spec, flatdim, global_head_state,
            lora_cfg, device, loss_type=cfg.train_loss,
        )
        projector = RademacherProjector(
            input_dim=flatdim,
            output_dim=cfg.gvendi_projection_dim,
            seed=cfg.gvendi_projection_seed,
        )
        manifest = run_selection(
            cfg, paths, engine, projector, server,
            active_clients, active_meta, active_features, feature_stats,
            base_ctor, base_state, lora_cfg, seq_len,
            n_regimes, prev_v, n_regimes, new_v,
            source_checkpoint=prev_ckpt_dir,
        )
        del engine

    if args.selection_only:
        print(f"[gvendi-vexp] --selection_only: manifest at "
              f"{paths.selection_manifest(method)}")
        return

    # ------------------------------------------------------------------
    # Build the full next-stage client set from the cumulative selection.
    # ------------------------------------------------------------------
    train_dir = paths.train_dir(method)
    os.makedirs(train_dir, exist_ok=True)

    final_metrics_path = os.path.join(
        train_dir, f"round_{new_round}_final_metrics.json"
    )
    if os.path.exists(final_metrics_path):
        print(f"[gvendi-vexp] {final_metrics_path} already exists; "
              f"nothing to do")
        return

    new_cum_rows = list(cum_rows) + list(manifest["selected_clients"])
    clients, meta_rows, client_features, new_feature_stats = (
        build_stage_clients_from_manifest(
            cfg, new_cum_rows, base_r, base_v, n_regimes, new_v,
            seq_len=seq_len, batch_size=cfg.batch_size,
        )
    )
    torch.save(new_feature_stats, os.path.join(train_dir, "client_feature_stats.pt"))
    atomic_write_json(
        os.path.join(train_dir, CUMULATIVE_FILENAME),
        {
            "base_n_regimes": base_r,
            "base_n_variants": base_v,
            "rows": new_cum_rows,
        },
    )

    cfg.synthetic_num_regimes = n_regimes
    cfg.synthetic_variants_per_regime = new_v
    cfg.synthetic_num_clients = len(clients)
    cfg.n_clients = len(clients)
    cfg.clients_per_round = min(cfg.clients_per_round, cfg.n_clients)

    dp_generators = (
        dp.load_or_init_generators(cfg, prev_ckpt_dir=prev_ckpt_dir)
        if cfg.gvendi_dp_mode != "off" else None
    )

    # Only THIS stage's additions get the sampling boost / new-client metrics
    # (clients added by earlier expansions are ordinary active clients now).
    new_client_ids = {
        cid for cid, m in enumerate(meta_rows)
        if int(m["variant_id"]) == prev_v
    }
    print(f"[gvendi-vexp] built {len(clients)} clients "
          f"({len(new_client_ids)} new, "
          f"{len(clients) - len(new_client_ids)} carried over)")

    # ------------------------------------------------------------------
    # Ordinary ReLoRA boundary (fresh optimizer: nothing to prune).
    # ------------------------------------------------------------------
    scheduler = CurriculumScheduler(cfg)
    target_lr = scheduler.get_target_lr(new_round)

    old_emb_param = server.emb.weight
    for group in server.opt.param_groups:
        group["lr"] = 0.0
    expand_server_embeddings(
        server, active_meta, meta_rows,
        old_n_regimes=n_regimes, old_n_variants=prev_v,
    )
    _rebuild_optimizer_with_new_emb(server, old_emb_param, weight_decay=1e-4)
    server.client_features = client_features.to(device)
    print(f"[gvendi-vexp] boundary applied: {client_features.shape[0]} clients, "
          f"target_lr={target_lr:.2e}, warmup={cfg.relora_lr_warmup_steps} rounds")

    regime_labels = torch.tensor(
        [int(row["regime_id"]) for row in meta_rows], dtype=torch.long
    )
    warmup_step_counter = [0]
    target_lrs_for_warmup = [target_lr] * len(server.opt.param_groups)
    patience_tracker = ClientPatienceTracker(
        patience=cfg.relora_client_patience,
        min_delta=cfg.relora_patience_min_delta,
    )

    def _jsonable(d):
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in d.items()
            if isinstance(v, (int, float, str, bool, list, tuple, type(None)))
        }

    atomic_write_json(
        os.path.join(train_dir, f"round_{new_round}_config.json"),
        {
            "curriculum_round": new_round,
            "n_regimes": n_regimes,
            "n_variants": new_v,
            "n_clients": cfg.n_clients,
            "target_lr": target_lr,
            "selection_method": method,
            **_jsonable(vars(cfg)),
        },
    )
    atomic_write_json(
        os.path.join(train_dir, f"round_{new_round}_client_meta.json"), meta_rows
    )

    # ------------------------------------------------------------------
    # Train the expanded stage.
    # ------------------------------------------------------------------
    csv_path = os.path.join(train_dir, "relora_history.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=GVENDI_CSV_FIELDNAMES)
        csv_writer.writeheader()
        final_metrics, global_head_state, _, _ = run_gvendi_stage(
            curriculum_round=new_round,
            cfg=cfg,
            server=server,
            clients=clients,
            meta_rows=meta_rows,
            base_ctor=base_ctor,
            base_state=base_state,
            spec=spec,
            global_head_state=global_head_state,
            device=device,
            regime_labels=regime_labels,
            patience_tracker=patience_tracker,
            scheduler=scheduler,
            warmup_step_counter=warmup_step_counter,
            target_lrs=target_lrs_for_warmup,
            csv_writer=csv_writer,
            csv_file=csv_file,
            max_comm_rounds=cfg.relora_max_rounds_per_curriculum,
            new_client_ids=new_client_ids,
            dp_generators=dp_generators,
        )

    # ------------------------------------------------------------------
    # Forgetting on the carried-over clients (vs. the source stage).
    # ------------------------------------------------------------------
    if final_metrics is not None:
        source_metrics_path = os.path.join(
            args.source_dir, f"round_{args.prev_round}_final_metrics.json"
        )
        if os.path.exists(source_metrics_path):
            with open(source_metrics_path) as f:
                source_metrics = json.load(f)
            before = {
                pc["regime_variant"]: pc.get("mase")
                for pc in source_metrics.get("per_client", [])
            }
            deltas = []
            for pc in final_metrics["per_client"]:
                key = pc.get("regime_variant")
                if (
                    pc["client_id"] not in new_client_ids
                    and key in before
                    and before[key] is not None
                    and pc.get("mase") is not None
                ):
                    deltas.append(float(pc["mase"]) - float(before[key]))
            if deltas:
                final_metrics["forgetting_mase"] = float(np.mean(deltas))
                print(f"[gvendi-vexp] forgetting (mean MASE delta on "
                      f"carried-over clients): "
                      f"{final_metrics['forgetting_mase']:+.5f}")
        atomic_write_json(final_metrics_path, final_metrics)

    # ------------------------------------------------------------------
    # Save final state (round-named so the next stage can chain on it).
    # ------------------------------------------------------------------
    last_ckpt_dir = os.path.join(train_dir, f"_relora_round_{new_round}_ckpt")
    os.makedirs(last_ckpt_dir, exist_ok=True)
    server.save(last_ckpt_dir)
    torch.save(global_head_state, os.path.join(last_ckpt_dir, "global_head_state.pt"))
    if cfg.gvendi_dp_mode != "off":
        dp.save_generator_state(dp_generators, last_ckpt_dir)
    torch.save(base_state, os.path.join(train_dir, "base_state.pt"))
    torch.save(
        {
            "spec": spec,
            "flatdim": flatdim,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "exclude_keywords": cfg.exclude_keywords,
        },
        os.path.join(train_dir, "lora_meta.pt"),
    )
    atomic_write_json(os.path.join(train_dir, "run_config.json"), _jsonable(vars(cfg)))
    print(f"\n[gvendi-vexp] done — outputs in {train_dir}/")


if __name__ == "__main__":
    main()
