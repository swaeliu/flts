"""Orchestrate H-Vendi-guided client expansion at a ReLoRA curriculum boundary.

Control flow (proposal section 13.8):

    load completed source run, restore stage-best checkpoint
    if valid selection manifest exists:
        load it (no re-selection)
    else:
        probe active clients post-restore
        load-or-generate candidate pool
        probe + filter candidates
        select clients (random | hard | h_vendi)
        write selection manifest atomically
    generate full next-stage clients from the manifest
    execute the ordinary ReLoRA boundary
    train the expanded stage

All three selection methods share the same candidate pool, probe gradients,
quality gate, quotas, sampling policy, and round budget — only the ranking
rule differs.

Usage
-----
    python -m gvendi.run_gvendi_8x8 \
        --out_dir <existing_run_dir> \
        --prev_round 5 \
        --prev_n_regimes 7 --prev_n_variants 7 \
        --new_n_regimes 8 --new_n_variants 8 \
        --gvendi_selection_method h_vendi \
        --relora_client_patience 40

Run with --selection_only for the Phase-1 offline selection prototype
(writes the manifest + report and stops before any training).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
for _sub in ("hypernet", "data_generation", "lora"):
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

from .candidates import (  # noqa: E402
    CandidateEvaluation,
    CandidateSpec,
    build_filter_report,
    evaluate_candidate_quality,
    generate_candidate_pool,
    promote_for_quota,
)
from .config import (  # noqa: E402
    GVENDI_OVERRIDE_NAMES,
    GVendiConfig,
    add_gvendi_cli_args,
    validate_gvendi_config,
)
from . import dp  # noqa: E402
from .data import build_stage_clients_from_manifest  # noqa: E402
from .grad_vendi import (  # noqa: E402
    EntropyState,
    GradVendi,
    RademacherProjector,
    SelectionCandidate,
    fit_active_gradient_clusters,
    select_existing_regime_candidates,
    select_hard,
    select_new_regime_candidates,
    select_random,
    shortlist_candidates,
)
from .manifest import (  # noqa: E402
    GVendiPaths,
    atomic_torch_save,
    atomic_write_json,
    build_selection_manifest,
    load_candidate_pool,
    save_candidate_pool,
    selected_client_row,
    try_load_selection_manifest,
)
from .probing import (  # noqa: E402
    ProbeEngine,
    collect_adapter_probe_gradients,
)
from .train_8x8 import GVENDI_CSV_FIELDNAMES, run_gvendi_stage  # noqa: E402


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_args():
    p = argparse.ArgumentParser(
        description="H-Vendi-guided client expansion for a ReLoRA run"
    )
    p.add_argument("--out_dir", type=str, required=True,
                   help="Existing ReLoRA run directory (source of the checkpoint)")
    p.add_argument("--prev_round", type=int, required=True,
                   help="Curriculum-round index whose checkpoint to expand from "
                        "(out_dir/_relora_round_{prev_round}_ckpt)")
    p.add_argument("--prev_n_regimes", type=int, required=True)
    p.add_argument("--prev_n_variants", type=int, required=True)
    p.add_argument("--new_n_regimes", type=int, required=True)
    p.add_argument("--new_n_variants", type=int, required=True)
    p.add_argument("--relora_client_patience", type=int, default=40)
    p.add_argument("--relora_max_rounds_per_curriculum", type=int, default=500)
    p.add_argument("--selection_only", action="store_true",
                   help="Phase 1: write the selection manifest + report and stop")
    add_gvendi_cli_args(p)
    return p.parse_args()


def _load_cfg(out_dir: str, prev_round: int) -> GVendiConfig:
    path = os.path.join(out_dir, f"round_{prev_round}_config.json")
    with open(path) as f:
        d = json.load(f)
    for admin_key in ("curriculum_round", "n_regimes", "n_variants", "n_clients", "target_lr"):
        d.pop(admin_key, None)
    cfg = GVendiConfig()
    for k, v in d.items():
        setattr(cfg, k, v)
    return cfg


def _build_base_model(cfg, device):
    from lora_utils import (
        get_lora_spec_and_flatdim,
        inject_lora,
        mark_only_lora_trainable,
    )

    backbone = getattr(cfg, "backbone", "moment")
    if backbone == "moment":
        from curriculum import train_relora as tr
        base_ctor = tr._make_base_model_ctor(cfg)
    elif backbone == "chronos":
        from gvendi_chronos.model_wrapper import make_base_model_ctor as _make_chronos_ctor
        base_ctor = _make_chronos_ctor(cfg)
    elif backbone == "patchtst":
        from gvendi_patchtst.model_wrapper import make_base_model_ctor as _make_patchtst_ctor
        base_ctor = _make_patchtst_ctor(cfg)
    else:
        raise ValueError(f"Unknown cfg.backbone: {backbone!r}")

    model0 = base_ctor().to(device)
    seq_len = model0.config.seq_len
    inject_lora(model0, r=cfg.lora_rank, alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout, exclude_keywords=cfg.exclude_keywords)
    mark_only_lora_trainable(model0)
    spec, flatdim = get_lora_spec_and_flatdim(model0)

    model_base = base_ctor().to("cpu")
    base_state = model_base.state_dict()
    del model_base, model0
    return base_ctor, base_state, spec, flatdim, seq_len


# ---------------------------------------------------------------------------
# Active-client probing (post-restore)
# ---------------------------------------------------------------------------

def probe_active_clients(
    cfg,
    paths: GVendiPaths,
    engine: ProbeEngine,
    projector: RademacherProjector,
    server,
    active_clients,
    active_meta,
):
    """Deterministic post-restore probe sweep over every active client.

    Returns (gradients: key -> [B, d], proxy_mase: key -> float). Cached in
    active_gradients.pt; the cache is invalidated when projector or probe
    settings change.
    """
    expected_meta = {
        "projector": projector.metadata(),
        "probe_seed": cfg.gvendi_probe_seed,
        "probe_batches": cfg.gvendi_probe_batches,
        "probe_batch_size": cfg.gvendi_probe_batch_size,
    }
    if cfg.gvendi_cache_gradients and os.path.exists(paths.active_gradients):
        payload = torch.load(paths.active_gradients, map_location="cpu",
                             weights_only=False)
        if payload.get("meta") == expected_meta:
            print(f"[gvendi] loaded cached active gradients ({paths.active_gradients})")
            return payload["gradients"], payload["proxy_mase"]
        print("[gvendi] active gradient cache stale (settings changed); re-probing")

    t0 = time.time()
    adapters_and_datasets = {}
    proxy_mase: dict[str, float] = {}
    for cid, meta in enumerate(active_meta):
        key = meta["regime_variant"]
        with torch.no_grad():
            flat = server.generate_lora_flat([cid])[0].detach().cpu()
        adapters_and_datasets[key] = (flat, active_clients[cid]["train"].dataset)
        mm = engine.evaluate(
            flat,
            active_clients[cid]["test"],
            max_batches=cfg.eval_batches,
            mase_denom=active_clients[cid].get("mase_denom"),
        )
        proxy_mase[key] = float(mm["mase"])

    gradients = collect_adapter_probe_gradients(
        engine,
        projector,
        adapters_and_datasets,
        probe_batches_count=cfg.gvendi_probe_batches,
        probe_batch_size=cfg.gvendi_probe_batch_size,
        probe_seed=cfg.gvendi_probe_seed,
    )
    print(f"[gvendi] probed {len(gradients)} active clients in {time.time()-t0:.1f}s")

    if cfg.gvendi_cache_gradients:
        atomic_torch_save(
            paths.active_gradients,
            {"meta": expected_meta, "gradients": gradients, "proxy_mase": proxy_mase},
        )
    return gradients, proxy_mase


# ---------------------------------------------------------------------------
# Candidate evaluation (probe + quality gate), with caching
# ---------------------------------------------------------------------------

def _eval_to_payload(ev: CandidateEvaluation) -> dict:
    return {
        "spec": ev.spec.to_dict(),
        "raw_feature_vector": ev.raw_feature_vector,
        "feature_vector": ev.feature_vector,
        "projected_gradients": ev.projected_gradients,
        "proxy_metrics": ev.proxy_metrics,
        "learnability_improvement": ev.learnability_improvement,
        "filter_results": ev.filter_results,
        "is_eligible": ev.is_eligible,
    }


def _eval_from_payload(d: dict) -> CandidateEvaluation:
    return CandidateEvaluation(
        spec=CandidateSpec.from_dict(d["spec"]),
        raw_feature_vector=d["raw_feature_vector"],
        feature_vector=d["feature_vector"],
        projected_gradients=d["projected_gradients"],
        proxy_metrics=d["proxy_metrics"],
        learnability_improvement=d["learnability_improvement"],
        filter_results=d["filter_results"],
        is_eligible=d["is_eligible"],
    )


def evaluate_candidates(
    cfg,
    paths: GVendiPaths,
    specs: list[CandidateSpec],
    engine: ProbeEngine,
    projector: RademacherProjector,
    adapter_fn,
    base_ctor,
    base_state,
    lora_cfg,
    feature_stats,
    mase_bounds,
    active_features_by_regime,
    seq_len: int,
) -> list[CandidateEvaluation]:
    """Probe and quality-gate every candidate, resuming from cache by id."""
    cached: dict[str, CandidateEvaluation] = {}
    if cfg.gvendi_cache_gradients and os.path.exists(paths.candidate_evals):
        payload = torch.load(paths.candidate_evals, map_location="cpu",
                             weights_only=False)
        if payload.get("mase_bounds") == list(mase_bounds):
            for d in payload["evals"]:
                ev = _eval_from_payload(d)
                cached[ev.spec.candidate_id] = ev
            print(f"[gvendi] loaded {len(cached)} cached candidate evaluations")
        else:
            print("[gvendi] candidate eval cache stale (MASE bounds changed)")

    evals: list[CandidateEvaluation] = []
    t0 = time.time()
    dirty = False
    for i, spec in enumerate(specs):
        if spec.candidate_id in cached:
            evals.append(cached[spec.candidate_id])
            continue
        ev = evaluate_candidate_quality(
            cfg,
            spec,
            engine,
            projector,
            adapter_fn,
            base_ctor,
            base_state,
            lora_cfg,
            feature_mean=feature_stats["mean"],
            feature_std=feature_stats["std"],
            mase_bounds=mase_bounds,
            existing_features_same_regime=active_features_by_regime.get(
                spec.regime_index, []
            ),
            seq_len=seq_len,
        )
        evals.append(ev)
        dirty = True
        if (i + 1) % 10 == 0:
            print(
                f"[gvendi] evaluated {i + 1}/{len(specs)} candidates "
                f"({time.time() - t0:.1f}s)"
            )
            if cfg.gvendi_cache_gradients:
                atomic_torch_save(
                    paths.candidate_evals,
                    {
                        "mase_bounds": list(mase_bounds),
                        "evals": [_eval_to_payload(e) for e in evals],
                    },
                )
                dirty = False

    if cfg.gvendi_cache_gradients and dirty:
        atomic_torch_save(
            paths.candidate_evals,
            {
                "mase_bounds": list(mase_bounds),
                "evals": [_eval_to_payload(e) for e in evals],
            },
        )
    return evals


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def run_selection(
    cfg,
    paths: GVendiPaths,
    engine: ProbeEngine,
    projector: RademacherProjector,
    server,
    active_clients,
    active_meta,
    active_features,
    feature_stats,
    base_ctor,
    base_state,
    lora_cfg,
    seq_len: int,
    prev_n_regimes: int,
    prev_n_variants: int,
    new_n_regimes: int,
    new_n_variants: int,
    source_checkpoint: str,
) -> dict:
    """Full selection pipeline; returns the (already persisted) manifest.

    Supports both expansion modes: +1 regime / +1 variant (a new regime is
    selected alongside one new variant per existing regime) and variant-only
    (new_n_regimes == prev_n_regimes; only the per-existing-regime picks).
    """
    method = cfg.gvendi_selection_method
    expand_regime = new_n_regimes > prev_n_regimes
    new_regime_index = prev_n_regimes
    t_start = time.time()

    # ---- Step 1: post-restore active sweep -------------------------------
    active_grads, active_mase = probe_active_clients(
        cfg, paths, engine, projector, server, active_clients, active_meta
    )
    t_probe_active = time.time() - t_start

    mase_values = np.array(sorted(active_mase.values()), dtype=np.float64)
    mase_bounds = (
        float(np.quantile(mase_values, cfg.gvendi_mase_low_quantile)),
        float(np.quantile(mase_values, cfg.gvendi_mase_high_quantile)),
    )
    print(f"[gvendi] active MASE band: [{mase_bounds[0]:.4f}, {mase_bounds[1]:.4f}]")

    # ---- Step 2: cluster the active gradient space -----------------------
    km = fit_active_gradient_clusters(
        active_grads,
        num_clusters=cfg.gvendi_num_clusters,
        sparse_cluster_fraction=cfg.gvendi_sparse_cluster_fraction,
        seed=cfg.gvendi_projection_seed + 1,
    )
    if cfg.gvendi_cache_gradients:
        atomic_torch_save(paths.kmeans_state, km.to_payload())

    # ---- Step 3: candidate pool (shared across methods) ------------------
    anchor_weights = {}
    for cid, meta in enumerate(active_meta):
        key = meta["regime_variant"]
        anchor_weights[(int(meta["regime_id"]), int(meta["variant_id"]))] = (
            km.active_sparse_fraction.get(key, 0.0)
        )

    if os.path.exists(paths.candidate_pool):
        specs = [CandidateSpec.from_dict(d)
                 for d in load_candidate_pool(paths.candidate_pool)]
        print(f"[gvendi] loaded candidate pool ({len(specs)} candidates)")
    else:
        specs = generate_candidate_pool(
            cfg,
            prev_n_regimes,
            prev_n_variants,
            new_n_regimes,
            new_regime_index,
            anchor_weights=anchor_weights,
            regimes=None if expand_regime else list(range(prev_n_regimes)),
        )
        save_candidate_pool(paths.candidate_pool, specs, cfg)
        print(f"[gvendi] generated candidate pool ({len(specs)} candidates)")

    # ---- Step 4: probe + quality-gate candidates -------------------------
    active_features_by_regime: dict[int, list[torch.Tensor]] = {}
    for cid, meta in enumerate(active_meta):
        active_features_by_regime.setdefault(int(meta["regime_id"]), []).append(
            active_features[cid]
        )

    def adapter_fn(feats: torch.Tensor) -> torch.Tensor:
        return server.generate_lora_flat_from_features(feats)

    evals = evaluate_candidates(
        cfg, paths, specs, engine, projector, adapter_fn, base_ctor, base_state,
        lora_cfg, feature_stats, mase_bounds, active_features_by_regime, seq_len,
    )

    # ---- Step 5: quotas — regenerate, then deterministic relaxation ------
    quotas = {r: 1 for r in range(prev_n_regimes)}
    if expand_regime:
        quotas[new_regime_index] = new_n_variants

    def _shortfall_regimes():
        out = []
        for regime, quota in quotas.items():
            n_ok = sum(
                1 for ev in evals
                if ev.spec.regime_index == regime and ev.is_eligible
            )
            if n_ok < quota:
                out.append(regime)
        return out

    shortfall = _shortfall_regimes()
    if shortfall:
        print(f"[gvendi] regimes under quota after filtering: {shortfall}; "
              f"generating one extra candidate batch")
        extra = generate_candidate_pool(
            cfg, prev_n_regimes, prev_n_variants, new_n_regimes,
            new_regime_index, anchor_weights=anchor_weights,
            extra_batch=1, regimes=shortfall,
        )
        specs = specs + extra
        save_candidate_pool(paths.candidate_pool, specs, cfg)
        evals = evaluate_candidates(
            cfg, paths, specs, engine, projector, adapter_fn, base_ctor,
            base_state, lora_cfg, feature_stats, mase_bounds,
            active_features_by_regime, seq_len,
        )
        promote_for_quota(evals, quotas)

    filter_report = build_filter_report(evals)
    atomic_write_json(paths.filter_report, filter_report)
    print(
        f"[gvendi] filter report: {filter_report['n_eligible']}/"
        f"{filter_report['n_proposed']} eligible"
    )

    # ---- Step 6: shortlist by sparse-region membership -------------------
    eligible = [ev for ev in evals if ev.is_eligible]
    sel_cands = [
        SelectionCandidate(
            candidate_id=ev.spec.candidate_id,
            regime_index=ev.spec.regime_index,
            gradients=ev.projected_gradients,
            feature_vector=ev.feature_vector,
            proxy_mase=float(ev.proxy_metrics.get("mase") or float("inf")),
        )
        for ev in eligible
    ]
    shortlist_candidates(
        sel_cands,
        km,
        min_sparse_fraction=cfg.gvendi_min_sparse_fraction,
        centroid_distance_quantile=cfg.gvendi_centroid_distance_quantile,
        required_per_regime=quotas,
    )
    by_id = {ev.spec.candidate_id: ev for ev in evals}

    def _pools(only_shortlisted: bool) -> dict[int, list[SelectionCandidate]]:
        pools: dict[int, list[SelectionCandidate]] = {}
        for c in sel_cands:
            if only_shortlisted and not c.shortlisted:
                continue
            pools.setdefault(c.regime_index, []).append(c)
        return pools

    # ---- Step 7: ranking rule (the only step that differs per method) ----
    active_matrix = torch.cat(
        [active_grads[k] for k in sorted(active_grads.keys())], dim=0
    )
    active_entropy = GradVendi.entropy(active_matrix)

    new_regime_quota = new_n_variants if expand_regime else 0
    if method == "h_vendi":
        pools = _pools(only_shortlisted=True)
        existing_picks, _ = select_existing_regime_candidates(
            active_matrix,
            {r: pools[r] for r in range(prev_n_regimes)},
            n_order_trials=cfg.gvendi_regime_order_trials,
            seed=cfg.gvendi_selection_seed,
        )
        if expand_regime:
            state = EntropyState(active_matrix)
            for c in existing_picks.values():
                state.add(c.gradients)
            new_picks = select_new_regime_candidates(
                state,
                pools[new_regime_index],
                quota=new_regime_quota,
                min_feature_distance=cfg.gvendi_min_selected_feature_distance,
            )
        else:
            new_picks = []
    elif method == "random":
        existing_picks, new_picks = select_random(
            _pools(only_shortlisted=True),
            new_regime_index,
            new_regime_quota,
            cfg.gvendi_min_selected_feature_distance,
            seed=cfg.gvendi_selection_seed,
        )
    elif method == "hard":
        existing_picks, new_picks = select_hard(
            _pools(only_shortlisted=False),
            new_regime_index,
            new_regime_quota,
            cfg.gvendi_min_selected_feature_distance,
        )
    else:
        raise ValueError(f"unknown selection method {method!r}")

    # ---- Step 8: assign grid cells + persist manifest --------------------
    selected_rows = []
    all_picks: list[SelectionCandidate] = []
    for regime in sorted(existing_picks.keys()):
        c = existing_picks[regime]
        selected_rows.append(
            selected_client_row(by_id[c.candidate_id], c, regime, prev_n_variants)
        )
        all_picks.append(c)
    for v, c in enumerate(new_picks):
        selected_rows.append(
            selected_client_row(by_id[c.candidate_id], c, new_regime_index, v)
        )
        all_picks.append(c)

    manifest = build_selection_manifest(
        cfg,
        method=method,
        source_stage=f"{prev_n_regimes}x{prev_n_variants}",
        target_stage=f"{new_n_regimes}x{new_n_variants}",
        source_checkpoint=source_checkpoint,
        selected_rows=selected_rows,
    )
    atomic_write_json(paths.selection_manifest(method), manifest)

    # ---- Step 9: human-readable selection report -------------------------
    post_state = EntropyState(active_matrix)
    for c in all_picks:
        post_state.add(c.gradients)
    post_entropy = post_state.entropy()

    selected_feats = [c.feature_vector for c in new_picks]
    pairwise = [
        float((selected_feats[i] - selected_feats[j]).norm().item())
        for i in range(len(selected_feats))
        for j in range(i + 1, len(selected_feats))
    ]

    report = {
        "selection_method": method,
        "active_h_vendi": float(np.exp(active_entropy)),
        "post_selection_h_vendi": float(np.exp(post_entropy)),
        "active_entropy": active_entropy,
        "post_selection_entropy": post_entropy,
        "entropy_increase_abs": post_entropy - active_entropy,
        "entropy_increase_rel": (
            (post_entropy - active_entropy) / active_entropy
            if active_entropy > 0 else None
        ),
        "cluster_occupancy": [int(v) for v in km.occupancy.tolist()],
        "sparse_clusters": list(km.sparse_clusters),
        "filter_report": filter_report,
        "selected_clients": selected_rows,
        "min_selected_pairwise_feature_distance": (
            min(pairwise) if pairwise else None
        ),
        "probe_wall_clock_secs_active": t_probe_active,
        "total_selection_wall_clock_secs": time.time() - t_start,
    }
    atomic_write_json(paths.selection_report(method), report)
    print(
        f"[gvendi] selection done ({method}): H-Vendi "
        f"{report['active_h_vendi']:.3f} -> {report['post_selection_h_vendi']:.3f}"
    )
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    cfg = _load_cfg(args.out_dir, args.prev_round)
    new_round = args.prev_round + 1
    method_args = {
        name: getattr(args, name)
        for name in GVENDI_OVERRIDE_NAMES
        if getattr(args, name, None) is not None
    }
    for name, val in method_args.items():
        setattr(cfg, name, val)

    cfg.relora_client_patience = args.relora_client_patience
    cfg.relora_max_rounds_per_curriculum = args.relora_max_rounds_per_curriculum
    cfg.relora_max_rounds_schedule = ""
    cfg.relora_curriculum_schedule = (
        cfg.relora_curriculum_schedule.rstrip(",")
        + f",{args.new_n_regimes}x{args.new_n_variants}"
    )

    validate_gvendi_config(
        cfg, args.prev_n_regimes, args.prev_n_variants,
        args.new_n_regimes, args.new_n_variants,
    )
    method = cfg.gvendi_selection_method

    device = _dev()
    print(f"[gvendi] device={device} method={method}")
    print(f"[gvendi] {args.prev_n_regimes}x{args.prev_n_variants} -> "
          f"{args.new_n_regimes}x{args.new_n_variants}")

    if not getattr(cfg, "use_fixed_client_features", False):
        raise NotImplementedError(
            "gvendi requires use_fixed_client_features=True (candidate "
            "adapters are generated from feature vectors)"
        )

    base_ctor, base_state, spec, flatdim, seq_len = _build_base_model(cfg, device)
    print(f"[gvendi] LoRA spec: {len(spec)} tensors, flatdim={flatdim:,}")
    lora_cfg = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "exclude_keywords": cfg.exclude_keywords,
    }

    # ------------------------------------------------------------------
    # Rebuild the active (previous-stage) client set, deterministically.
    # ------------------------------------------------------------------
    cfg.synthetic_num_regimes = args.prev_n_regimes
    cfg.synthetic_variants_per_regime = args.prev_n_variants
    cfg.synthetic_num_clients = args.prev_n_regimes * args.prev_n_variants
    cfg.n_clients = cfg.synthetic_num_clients
    cfg.data_source = "synthetic"

    set_seed(cfg.seed)
    active_clients, active_meta, active_features, feature_stats = (
        make_synthetic_clients(
            cfg, seq_len=seq_len, batch_size=cfg.batch_size,
            return_feature_stats=True,
        )
    )
    feat_dim = int(active_features.shape[1])
    if cfg.emb_dim != feat_dim:
        cfg.emb_dim = feat_dim

    # ------------------------------------------------------------------
    # Restore the stage-best checkpoint and freeze it for selection.
    # ------------------------------------------------------------------
    prev_ckpt_dir = os.path.join(
        args.out_dir, f"_relora_round_{args.prev_round}_ckpt"
    )
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
    print(f"[gvendi] restored stage-best state from {prev_ckpt_dir}")

    # ------------------------------------------------------------------
    # Selection (resume-safe): skip entirely when a valid manifest exists.
    # ------------------------------------------------------------------
    paths = GVendiPaths(args.out_dir, new_round).ensure()
    manifest = None
    if cfg.gvendi_resume_from_manifest:
        manifest = try_load_selection_manifest(
            paths.selection_manifest(method), method,
            args.prev_n_regimes, args.prev_n_variants,
            args.new_n_regimes, args.new_n_variants,
        )
        if manifest is not None:
            print(f"[gvendi] valid selection manifest found; skipping selection")

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
            args.prev_n_regimes, args.prev_n_variants,
            args.new_n_regimes, args.new_n_variants,
            source_checkpoint=prev_ckpt_dir,
        )
        del engine

    if args.selection_only:
        print(f"[gvendi] --selection_only: manifest at "
              f"{paths.selection_manifest(method)}")
        return

    # ------------------------------------------------------------------
    # Build the full next-stage client set from the manifest.
    # ------------------------------------------------------------------
    train_dir = paths.train_dir(method)
    os.makedirs(train_dir, exist_ok=True)

    final_metrics_path = os.path.join(
        train_dir, f"round_{new_round}_final_metrics.json"
    )
    if os.path.exists(final_metrics_path):
        print(f"[gvendi] {final_metrics_path} already exists; nothing to do")
        return

    clients, meta_rows, client_features, new_feature_stats = (
        build_stage_clients_from_manifest(
            cfg,
            manifest["selected_clients"],
            args.prev_n_regimes, args.prev_n_variants,
            args.new_n_regimes, args.new_n_variants,
            seq_len=seq_len,
            batch_size=cfg.batch_size,
        )
    )
    torch.save(new_feature_stats, os.path.join(train_dir, "client_feature_stats.pt"))

    cfg.synthetic_num_regimes = args.new_n_regimes
    cfg.synthetic_variants_per_regime = args.new_n_variants
    cfg.synthetic_num_clients = len(clients)
    cfg.n_clients = len(clients)
    cfg.clients_per_round = min(cfg.clients_per_round, cfg.n_clients)

    dp_generators = (
        dp.load_or_init_generators(cfg, prev_ckpt_dir=prev_ckpt_dir)
        if cfg.gvendi_dp_mode != "off" else None
    )

    new_client_ids = {
        cid for cid, m in enumerate(meta_rows) if m.get("is_new_client")
    }
    print(f"[gvendi] built {len(clients)} clients "
          f"({len(new_client_ids)} new, {len(clients) - len(new_client_ids)} original)")

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
        old_n_regimes=args.prev_n_regimes, old_n_variants=args.prev_n_variants,
    )
    _rebuild_optimizer_with_new_emb(server, old_emb_param, weight_decay=1e-4)
    server.client_features = client_features.to(device)
    print(f"[gvendi] boundary applied: {client_features.shape[0]} clients, "
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

    def _save_json(path, obj):
        atomic_write_json(path, obj)

    def _jsonable(d):
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in d.items()
            if isinstance(v, (int, float, str, bool, list, tuple, type(None)))
        }

    _save_json(
        os.path.join(train_dir, f"round_{new_round}_config.json"),
        {
            "curriculum_round": new_round,
            "n_regimes": args.new_n_regimes,
            "n_variants": args.new_n_variants,
            "n_clients": cfg.n_clients,
            "target_lr": target_lr,
            "selection_method": method,
            **_jsonable(vars(cfg)),
        },
    )
    _save_json(
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
    # Forgetting on the original clients (vs. the source stage's metrics).
    # ------------------------------------------------------------------
    if final_metrics is not None:
        source_metrics_path = os.path.join(
            args.out_dir, f"round_{args.prev_round}_final_metrics.json"
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
                print(f"[gvendi] forgetting (mean MASE delta on originals): "
                      f"{final_metrics['forgetting_mase']:+.5f}")
        _save_json(final_metrics_path, final_metrics)

    # ------------------------------------------------------------------
    # Save final state.
    # ------------------------------------------------------------------
    last_ckpt_dir = os.path.join(train_dir, "_relora_final_ckpt")
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
    _save_json(os.path.join(train_dir, "run_config.json"), _jsonable(vars(cfg)))
    print(f"\n[gvendi] done — outputs in {train_dir}/")


if __name__ == "__main__":
    main()
