"""Post-expansion stage training with new-client sampling boost + split metrics.

This adapts relora.train_relora._run_one_curriculum_round without modifying
it: client sampling is weighted (new clients get a decaying multiplier and a
per-batch occupancy cap — identical across Random / Hard / H-Vendi, it is not
part of any ranking rule), and the history CSV gains old/new-client metric
columns. The server update still regresses onto updated_flat (h + delta), not
the adapter displacement.
"""

from __future__ import annotations

import os
import sys
import time
from statistics import mean

import numpy as np
import torch

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
for _sub in ("clients", "lora"):
    _p = os.path.join(_SRC_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from client import local_train_lora_and_head_steps  # noqa: E402

from curriculum import train_relora as tr  # noqa: E402

from . import dp  # noqa: E402

GVENDI_CSV_FIELDNAMES = [
    "curriculum_round", "comm_round",
    "server_target_loss", "server_total_loss",
    "mean_client_mae", "mean_client_mse", "mean_client_smape",
    "mean_client_mase", "mean_relative_mse", "mean_client_val_mae",
    "mean_delta_norm",
    "patience_converged", "patience_sampled", "convergence_frac",
    "secs_round", "n_clients",
    # gvendi additions
    "old_client_mean_mae", "new_client_mean_mae",
    "old_client_mean_mase", "new_client_mean_mase",
    "worst_regime_mase", "new_client_sampling_multiplier",
    # FedChronos-style DP-perturbation diagnostics (None unless
    # gvendi_dp_mode != "off"; see gvendi/dp.py)
    "dp_clip_fraction", "dp_sigma", "dp_epsilon",
    "mean_clipped_delta_norm", "mean_noise_norm",
    "mean_noised_delta_norm", "mean_noise_to_signal_ratio",
]


# ---------------------------------------------------------------------------
# New-client sampling policy (identical across selection methods)
# ---------------------------------------------------------------------------

def new_client_weight(round_in_stage: int, multiplier: float, boost_rounds: int) -> float:
    """w_new(t) = 1 + (alpha - 1) * max(0, 1 - t / T_boost), t 0-indexed."""
    if boost_rounds <= 0:
        return 1.0
    return 1.0 + (multiplier - 1.0) * max(0.0, 1.0 - round_in_stage / boost_rounds)

def sample_clients_weighted(
    rng: np.random.Generator,
    n_clients: int,
    new_client_ids: set[int],
    k: int,
    new_weight: float,
    max_new_fraction: float,
) -> list[int]:
    """Weighted sampling without replacement, with a cap on new-client slots."""
    k = min(k, n_clients)
    weights = np.ones(n_clients, dtype=np.float64)
    for cid in new_client_ids:
        if cid < n_clients:
            weights[cid] = new_weight
    probs = weights / weights.sum()
    picked = list(rng.choice(n_clients, size=k, replace=False, p=probs))
    picked = [int(c) for c in picked]

    cap = int(np.floor(max_new_fraction * k))
    new_picked = [c for c in picked if c in new_client_ids]
    if len(new_picked) > cap:
        old_pool = [
            c for c in range(n_clients)
            if c not in new_client_ids and c not in picked
        ]
        # Under (near-)full participation, or whenever there simply aren't
        # enough unpicked old clients, the cap cannot be honored -- swap in
        # as many replacements as actually exist and leave the rest, rather
        # than crashing on an empty/undersized old_pool.
        excess = min(len(new_picked) - cap, len(old_pool))
        if excess > 0:
            drop = list(rng.choice(len(new_picked), size=excess, replace=False))
            drop_ids = {new_picked[i] for i in drop}
            replacements = list(rng.choice(len(old_pool), size=excess, replace=False))
            repl_iter = iter(replacements)
            picked = [
                c if c not in drop_ids else int(old_pool[next(repl_iter)])
                for c in picked
            ]
    return picked


# ---------------------------------------------------------------------------
# Old/new split metrics
# ---------------------------------------------------------------------------

def split_metrics(per_client: list[dict], new_client_ids: set[int]) -> dict:
    old_mae, new_mae, old_mase, new_mase = [], [], [], []
    regime_mase: dict[int, list[float]] = {}
    for pc in per_client:
        is_new = pc["client_id"] in new_client_ids
        (new_mae if is_new else old_mae).append(pc["mae"])
        if pc.get("mase") is not None:
            (new_mase if is_new else old_mase).append(pc["mase"])
            regime_mase.setdefault(int(pc["regime_id"]), []).append(pc["mase"])
    return {
        "old_client_mean_mae": float(mean(old_mae)) if old_mae else None,
        "new_client_mean_mae": float(mean(new_mae)) if new_mae else None,
        "old_client_mean_mase": float(mean(old_mase)) if old_mase else None,
        "new_client_mean_mase": float(mean(new_mase)) if new_mase else None,
        "worst_regime_mase": (
            float(max(mean(v) for v in regime_mase.values()))
            if regime_mase else None
        ),
    }


# ---------------------------------------------------------------------------
# Stage loop
# ---------------------------------------------------------------------------

def run_gvendi_stage(
    curriculum_round: int,
    cfg,
    server,
    clients: list[dict],
    meta_rows: list[dict],
    base_ctor,
    base_state: dict,
    spec,
    global_head_state: dict,
    device,
    regime_labels: torch.Tensor,
    patience_tracker,
    scheduler,
    warmup_step_counter: list[int],
    target_lrs: list[float],
    csv_writer,
    csv_file,
    max_comm_rounds: int,
    new_client_ids: set[int],
    track_checkpoint_variants: bool = False,
    dp_generators: dict[int, torch.Generator] | None = None,
):
    """Train the expanded stage until per-client convergence or the round cap.

    dp_generators: per-client torch.Generator for the FedChronos-style DP
    perturbation (gvendi/dp.py), keyed by client id. Ignored unless
    cfg.gvendi_dp_mode != "off". Callers that want the noise stream to
    continue across curriculum stages (rather than restart fresh every
    stage) must build this via dp.load_or_init_generators(cfg,
    prev_ckpt_dir) and persist it via dp.save_generator_state after the
    call returns -- see run_gvendi_8x8.main for the reference wiring. If
    left None and DP is active, ephemeral generators are seeded fresh
    from cfg.gvendi_dp_seed for this call only (no cross-stage
    continuation).

    Returns (final_metrics, global_head_state, last_row, checkpoint_variants).
    checkpoint_variants is None unless track_checkpoint_variants is True, in
    which case it is a dict with 5 keys, each a
    {"hnet", "emb", "head", "metrics", "comm_round"} snapshot:
      - "global_best": the restore-best snapshot behavior, unchanged --
        minimizes the population-blended mean_client_val_mae. As the
        old:new client ratio grows across stages this criterion is
        increasingly dominated by the (already-converged) old cohort.
      - "new_best": minimizes new-cohort-only mean val_mae, ignoring old
        clients entirely (the opposite extreme from global_best).
      - "cohort_balanced_best": minimizes the unweighted mean of
        (old-cohort mean val_mae, new-cohort mean val_mae) -- each cohort
        gets equal say regardless of its population size, directly
        cancelling the imbalance that biases global_best.
      - "constrained_best": minimizes new-cohort mean val_mae subject to
        old-cohort mean val_mae staying within
        (1 + cfg.gvendi_forgetting_tolerance) of the best old-cohort value
        seen so far in the stage (a running/causal bound, not a
        full-trajectory retrospective optimum -- old clients converge
        early and stay near-flat in practice, so this tracks the true
        constrained optimum closely without a second pass or storing
        every round's state).
      - "final": the last evaluated round's live state, pre any restore.
    Adds negligible overhead (a few extra in-memory state_dict copies) and
    does not change server's returned state, which is still the restored
    global-best when relora_restore_best is set, exactly as before.
    """
    lora_cfg = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "exclude_keywords": cfg.exclude_keywords,
    }
    sampling_rng = np.random.default_rng(cfg.seed + 31_337 + curriculum_round)

    dp_active = cfg.gvendi_dp_mode != "off"
    dp_clip_norm, dp_noise_std = dp.resolve_dp_params(cfg)
    dp_sigma = (
        dp.sigma_from_epsilon(cfg.gvendi_dp_epsilon, cfg.gvendi_dp_delta)
        if dp_active else None
    )
    if dp_active and dp_generators is None:
        dp_generators = dp.load_or_init_generators(cfg, prev_ckpt_dir=None)

    final_metrics = None
    last_row: dict = {}

    best_val_mae = float("inf")
    best_snapshot: dict | None = None
    best_metrics: dict | None = None

    new_best_val_mae = float("inf")
    new_best_snapshot: dict | None = None
    new_best_metrics: dict | None = None

    cohort_balanced_best_val = float("inf")
    cohort_balanced_best_snapshot: dict | None = None
    cohort_balanced_best_metrics: dict | None = None

    running_best_old_val_mae = float("inf")
    constrained_best_new_val_mae = float("inf")
    constrained_best_snapshot: dict | None = None
    constrained_best_metrics: dict | None = None

    warmup_rounds_this_stage = max(
        cfg.relora_lr_warmup_steps - warmup_step_counter[0], 0
    )

    for t in range(1, max_comm_rounds + 1):
        t0 = time.time()
        w_new = new_client_weight(
            t - 1,
            cfg.gvendi_new_client_sampling_multiplier,
            cfg.gvendi_new_client_sampling_rounds,
        )
        client_ids = sample_clients_weighted(
            sampling_rng,
            cfg.n_clients,
            new_client_ids,
            cfg.clients_per_round,
            w_new,
            cfg.gvendi_new_client_batch_fraction,
        )

        adapter_targets: dict[int, torch.Tensor] = {}
        delta_norms: list[float] = []
        local_head_states: list[dict] = []
        clipped_norms: list[float] = []
        noise_norms: list[float] = []
        noised_norms: list[float] = []
        noise_ratios: list[float] = []
        dp_clipped_count = 0

        for cid in client_ids:
            with torch.no_grad():
                init_flat = server.generate_lora_flat([cid])[0].detach().cpu()

            updated_flat, updated_head = local_train_lora_and_head_steps(
                base_model_ctor=base_ctor,
                base_state_dict=base_state,
                spec=spec,
                init_lora_flat=init_flat,
                init_head_state=global_head_state,
                train_loader=clients[cid]["train"],
                device=device,
                local_steps=cfg.local_steps,
                lr=cfg.local_lr,
                lora_cfg=lora_cfg,
                loss_type=cfg.train_loss,
            )
            # Regression target is h + delta (updated_flat), matching the
            # adapter-target objective used everywhere else -- optionally
            # clipped and noised (FedChronos-style; LoRA update only, see
            # gvendi/dp.py) before being handed to the server.
            if dp_active:
                raw_delta = updated_flat - init_flat
                noised_delta, diag = dp.clip_and_noise_delta(
                    raw_delta, dp_clip_norm, dp_noise_std, dp_generators[cid]
                )
                adapter_targets[cid] = init_flat + noised_delta
                delta_norms.append(diag["raw_norm"])
                clipped_norms.append(diag["clipped_norm"])
                noise_norms.append(diag["noise_norm"])
                noised_norms.append(diag["noised_norm"])
                noise_ratios.append(diag["noise_signal_ratio"])
                if dp_clip_norm is not None and diag["raw_norm"] > dp_clip_norm:
                    dp_clipped_count += 1
            else:
                adapter_targets[cid] = updated_flat
                delta_norms.append(float((updated_flat - init_flat).norm().item()))
            local_head_states.append(updated_head)

        server_losses = server.update_from_adapters(
            client_ids,
            adapter_targets,
            contrastive_labels=regime_labels[client_ids],
            use_contrastive_loss=cfg.use_contrastive_loss,
            contrastive_weight=cfg.contrastive_weight,
            contrastive_temperature=cfg.contrastive_temperature,
            contrastive_mode=cfg.contrastive_mode,
            return_details=True,
        )
        global_head_state = tr._average_state_dicts(local_head_states)

        ws = warmup_step_counter[0]
        if scheduler.is_in_warmup(ws):
            scale = scheduler.apply_warmup(server.opt, ws, target_lrs)
            print(
                f"[warmup] step {ws}/{cfg.relora_lr_warmup_steps} "
                f"scale={scale:.4f} lr={server.opt.param_groups[0]['lr']:.2e}"
            )
        elif cfg.relora_lr_schedule == "cosine":
            scheduler.apply_cosine_decay(
                server.opt, t, warmup_rounds_this_stage, max_comm_rounds, target_lrs
            )
        warmup_step_counter[0] += 1

        if cfg.relora_patience_scope == "sampled":
            for cid in client_ids:
                val_loss = tr._eval_client_val_loss(
                    server, clients[cid], cid, cfg, base_ctor, base_state,
                    spec, global_head_state, device,
                )
                _, converged = patience_tracker.update(cid, val_loss)
                if converged:
                    print(f"[patience] client {cid} converged")

        secs = time.time() - t0
        do_eval = (t % cfg.eval_every) == 0 or t == 1 or t == max_comm_rounds

        dp_row_fields = {
            "dp_clip_fraction": (
                dp_clipped_count / len(client_ids)
            ) if dp_active and client_ids else None,
            "dp_sigma": float(dp_sigma) if dp_active else None,
            "dp_epsilon": float(cfg.gvendi_dp_epsilon) if dp_active else None,
            "mean_clipped_delta_norm": (
                float(mean(clipped_norms)) if dp_active and clipped_norms else None
            ),
            "mean_noise_norm": (
                float(mean(noise_norms)) if dp_active and noise_norms else None
            ),
            "mean_noised_delta_norm": (
                float(mean(noised_norms)) if dp_active and noised_norms else None
            ),
            "mean_noise_to_signal_ratio": (
                float(mean(noise_ratios)) if dp_active and noise_ratios else None
            ),
        }

        row = {
            "curriculum_round": curriculum_round,
            "comm_round": t,
            "server_target_loss": float(server_losses["server_target_loss"]),
            "server_total_loss": float(server_losses["server_total_loss"]),
            "mean_client_mae": None,
            "mean_client_mse": None,
            "mean_client_smape": None,
            "mean_client_mase": None,
            "mean_relative_mse": None,
            "mean_client_val_mae": None,
            "mean_delta_norm": float(mean(delta_norms)) if delta_norms else 0.0,
            "patience_converged": patience_tracker.n_converged,
            "patience_sampled": patience_tracker.n_sampled,
            "convergence_frac": patience_tracker.n_converged / max(cfg.n_clients, 1),
            "secs_round": float(secs),
            "n_clients": cfg.n_clients,
            "old_client_mean_mae": None,
            "new_client_mean_mae": None,
            "old_client_mean_mase": None,
            "new_client_mean_mase": None,
            "worst_regime_mase": None,
            "new_client_sampling_multiplier": float(w_new),
            **dp_row_fields,
        }

        if do_eval:
            metrics = tr._evaluate_all_clients(
                server, clients, cfg, base_ctor, base_state, spec,
                global_head_state, device, meta_rows, None, None,
            )
            if cfg.relora_patience_scope != "sampled":
                for pc in metrics["per_client"]:
                    _, converged = patience_tracker.update(
                        pc["client_id"], pc["val_mae"]
                    )
                    if converged:
                        print(
                            f"[patience] client {pc['client_id']} converged "
                            f"(best val_mae="
                            f"{patience_tracker._best_loss[pc['client_id']]:.6f})"
                        )

            split = split_metrics(metrics["per_client"], new_client_ids)
            final_metrics = dict(metrics)
            final_metrics.update(split)
            final_metrics.update(
                {
                    "curriculum_round": curriculum_round,
                    "comm_round": t,
                    "server_target_loss": float(server_losses["server_target_loss"]),
                    "server_total_loss": float(server_losses["server_total_loss"]),
                    "mean_delta_norm": float(mean(delta_norms)) if delta_norms else 0.0,
                    "new_client_sampling_multiplier": float(w_new),
                    **dp_row_fields,
                }
            )

            if cfg.relora_restore_best and metrics["mean_client_val_mae"] < best_val_mae:
                best_val_mae = metrics["mean_client_val_mae"]
                best_snapshot = {
                    "hnet": {k: v.detach().cpu().clone()
                             for k, v in server.hnet.state_dict().items()},
                    "emb": {k: v.detach().cpu().clone()
                            for k, v in server.emb.state_dict().items()},
                    "head": {k: v.clone() for k, v in global_head_state.items()},
                    "comm_round": t,
                }
                best_metrics = dict(final_metrics)

            if track_checkpoint_variants:
                old_val = [
                    pc["val_mae"] for pc in metrics["per_client"]
                    if pc["client_id"] not in new_client_ids
                    and pc.get("val_mae") is not None
                ]
                new_val = [
                    pc["val_mae"] for pc in metrics["per_client"]
                    if pc["client_id"] in new_client_ids
                    and pc.get("val_mae") is not None
                ]

                def _snapshot():
                    return {
                        "hnet": {k: v.detach().cpu().clone()
                                 for k, v in server.hnet.state_dict().items()},
                        "emb": {k: v.detach().cpu().clone()
                                for k, v in server.emb.state_dict().items()},
                        "head": {k: v.clone() for k, v in global_head_state.items()},
                        "comm_round": t,
                    }

                if new_val:
                    new_client_val_mae = float(mean(new_val))
                    if new_client_val_mae < new_best_val_mae:
                        new_best_val_mae = new_client_val_mae
                        new_best_snapshot = _snapshot()
                        new_best_metrics = dict(final_metrics)
                        new_best_metrics["new_client_val_mae"] = new_client_val_mae

                if old_val and new_val:
                    old_client_val_mae = float(mean(old_val))

                    # cohort_balanced_best: equal weight per cohort, cancels
                    # the population-size imbalance in mean_client_val_mae.
                    cohort_balanced_val = 0.5 * (old_client_val_mae + new_client_val_mae)
                    if cohort_balanced_val < cohort_balanced_best_val:
                        cohort_balanced_best_val = cohort_balanced_val
                        cohort_balanced_best_snapshot = _snapshot()
                        cohort_balanced_best_metrics = dict(final_metrics)
                        cohort_balanced_best_metrics.update({
                            "old_client_val_mae": old_client_val_mae,
                            "new_client_val_mae": new_client_val_mae,
                            "cohort_balanced_val_mae": cohort_balanced_val,
                        })

                    # constrained_best: minimize new-cohort val_mae subject to
                    # old-cohort val_mae staying within a bounded regression
                    # of the best old-cohort value seen so far (causal, not a
                    # full-trajectory retrospective optimum -- see docstring).
                    running_best_old_val_mae = min(
                        running_best_old_val_mae, old_client_val_mae
                    )
                    threshold = running_best_old_val_mae * (
                        1.0 + cfg.gvendi_forgetting_tolerance
                    )
                    if (
                        old_client_val_mae <= threshold
                        and new_client_val_mae < constrained_best_new_val_mae
                    ):
                        constrained_best_new_val_mae = new_client_val_mae
                        constrained_best_snapshot = _snapshot()
                        constrained_best_metrics = dict(final_metrics)
                        constrained_best_metrics.update({
                            "old_client_val_mae": old_client_val_mae,
                            "new_client_val_mae": new_client_val_mae,
                            "forgetting_threshold_val_mae": threshold,
                        })

            row.update(
                {
                    "mean_client_mae": float(metrics["mean_client_mae"]),
                    "mean_client_mse": float(metrics["mean_client_mse"]),
                    "mean_client_smape": float(metrics["mean_client_smape"]),
                    "mean_client_mase": float(metrics["mean_client_mase"])
                        if metrics["mean_client_mase"] is not None else None,
                    "mean_relative_mse": None,
                    "mean_client_val_mae": float(metrics["mean_client_val_mae"]),
                    "patience_converged": patience_tracker.n_converged,
                    "convergence_frac": patience_tracker.n_converged / max(cfg.n_clients, 1),
                    **split,
                }
            )
            print(
                f"[gvendi crnd {curriculum_round} round {t:04d}] "
                f"mae={metrics['mean_client_mae']:.5f} "
                f"old_mae={split['old_client_mean_mae']} "
                f"new_mae={split['new_client_mean_mae']} "
                f"converged={patience_tracker.n_converged}/{cfg.n_clients} "
                f"secs={secs:.1f}"
            )
        else:
            print(
                f"[gvendi crnd {curriculum_round} round {t:04d}] "
                f"train_loss={float(server_losses['server_target_loss']):.5f} "
                f"w_new={w_new:.2f} secs={secs:.1f}"
            )

        last_row = row
        csv_writer.writerow(row)
        csv_file.flush()

        if patience_tracker.all_converged(cfg.n_clients):
            print(
                f"[gvendi] stage {curriculum_round} complete: "
                f"all {cfg.n_clients} clients exhausted patience"
            )
            break

    checkpoint_variants = None
    if track_checkpoint_variants:
        # Capture the live (pre-restore) state as "final" before any restore
        # below overwrites server.hnet/emb/global_head_state.
        final_snapshot = {
            "hnet": {k: v.detach().cpu().clone()
                     for k, v in server.hnet.state_dict().items()},
            "emb": {k: v.detach().cpu().clone()
                    for k, v in server.emb.state_dict().items()},
            "head": {k: v.clone() for k, v in global_head_state.items()},
            "comm_round": last_row.get("comm_round"),
        }
        checkpoint_variants = {
            "global_best": {
                **(best_snapshot or {}),
                "metrics": best_metrics,
            } if best_snapshot is not None else None,
            "new_best": {
                **(new_best_snapshot or {}),
                "metrics": new_best_metrics,
            } if new_best_snapshot is not None else None,
            "cohort_balanced_best": {
                **(cohort_balanced_best_snapshot or {}),
                "metrics": cohort_balanced_best_metrics,
            } if cohort_balanced_best_snapshot is not None else None,
            "constrained_best": {
                **(constrained_best_snapshot or {}),
                "metrics": constrained_best_metrics,
            } if constrained_best_snapshot is not None else None,
            "final": {**final_snapshot, "metrics": final_metrics},
        }

    if cfg.relora_restore_best and best_snapshot is not None:
        server.hnet.load_state_dict(best_snapshot["hnet"])
        server.emb.load_state_dict(best_snapshot["emb"])
        global_head_state = best_snapshot["head"]
        if best_metrics is not None:
            final_metrics = dict(best_metrics)
            final_metrics["restored_best_round"] = best_snapshot["comm_round"]
        print(
            f"[gvendi] stage {curriculum_round}: restored best state from "
            f"round {best_snapshot['comm_round']} (mean val MAE {best_val_mae:.5f})"
        )

    return final_metrics, global_head_state, last_row, checkpoint_variants
