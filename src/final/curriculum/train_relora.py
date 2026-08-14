"""ReLoRA curriculum training script.

Wraps the synthetic_fed_hnet_lora training loop with:
  - multi-round curriculum (expanding regimes + variants each round)
  - per-client patience tracking
  - magnitude-based Adam reset at round boundaries
  - LR warmup after each reset

Usage
-----
    python src/relora/train_relora.py \
        --out_dir runs/relora_5x5_to_7x7 \
        --relora_num_rounds 3 \
        --relora_client_patience 10

All flags from run_federated.py are accepted; new ReLoRA flags are listed below.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from statistics import mean, median, pstdev

import torch

# Make the split-out config/lora/clients/hypernet/data_generation packages
# importable as bare modules (config, lora_utils, client, server, synthetic_data).
_ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
for _sub in ("config", "lora", "clients", "hypernet", "data_generation"):
    _p = os.path.join(_ROOT_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from momentfm import MOMENTPipeline

from config import Config as _BaseConfig  # noqa: F401
from lora_utils import (
    flatten_lora,
    get_lora_spec_and_flatdim,
    inject_lora,
    load_flat_lora_into_model,
    mark_only_lora_trainable,
)
from client import (
    evaluate_forecast,
    extract_forecast_head_state_dict,
    load_forecast_head_state_dict,
    local_train_lora_and_head_steps,
)
from server import Server
from synthetic_data import make_synthetic_clients, set_seed

from .config import ReLoRAConfig
from .grad_variance import GradVarianceTracker
from .patience import ClientPatienceTracker
from .curriculum import CurriculumScheduler, build_key_to_index


# ---------------------------------------------------------------------------
# Helpers shared with run_federated.py (copied to avoid circular imports)
# ---------------------------------------------------------------------------

def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_base_model_ctor(cfg: ReLoRAConfig):
    def ctor():
        pipe = MOMENTPipeline.from_pretrained(
            cfg.model_id,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": cfg.horizon,
                "n_channels": cfg.n_channels,
            },
        )
        pipe.init()
        for p in pipe.parameters():
            p.requires_grad = False
        return pipe

    return ctor


def _average_state_dicts(state_dict_list: list[dict]) -> dict:
    if not state_dict_list:
        raise ValueError("state_dict_list is empty")
    out = {}
    for k in state_dict_list[0].keys():
        vals = [sd[k].float() for sd in state_dict_list]
        out[k] = torch.stack(vals, dim=0).mean(0).cpu()
    return out


def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _evaluate_all_clients(
    server,
    clients: list[dict],
    cfg: ReLoRAConfig,
    base_ctor,
    base_state: dict,
    spec,
    global_head_state: dict,
    device,
    meta_rows: list[dict],
    frozen_mse_by_client: dict | None = None,
    frozen_mae_by_client: dict | None = None,
) -> dict:
    """Evaluate every client on its test and val splits; return aggregated metrics.

    The val-split MAE rides on the same per-client model build as the test
    eval, so it costs one extra forward pass set rather than a second model
    construction. It feeds per-client patience (relora_patience_scope="all")
    and stage-best checkpoint selection (relora_restore_best).
    """
    per_client = []
    client_mse, client_mae, client_smape, client_mase, rel_mse = [], [], [], [], []
    client_val_mae = []
    total_se, total_ae, total_smape_sum, total_n = 0.0, 0.0, 0.0, 0

    for cid in range(cfg.n_clients):
        with torch.no_grad():
            flat = server.generate_lora_flat([cid])[0].detach().cpu()

        model = base_ctor().to(device)
        model.load_state_dict(base_state, strict=True)
        load_forecast_head_state_dict(model, global_head_state)
        inject_lora(model, r=cfg.lora_rank, alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout, exclude_keywords=cfg.exclude_keywords)
        mark_only_lora_trainable(model)
        load_flat_lora_into_model(model, spec, flat)

        mm = evaluate_forecast(
            model,
            clients[cid]["test"],
            device,
            max_batches=cfg.eval_batches,
            mase_denom=clients[cid].get("mase_denom"),
            seasonal_naive_mape=clients[cid].get("seasonal_naive_mape"),
        )

        vv = evaluate_forecast(
            model,
            clients[cid]["val"],
            device,
            max_batches=cfg.eval_batches,
        )
        client_val_mae.append(float(vv["mae"]))

        total_se += mm["se_sum"]
        total_ae += mm["ae_sum"]
        total_smape_sum += mm["smape_sum"]
        total_n += mm["n"]

        client_mse.append(mm["mse"])
        client_mae.append(mm["mae"])
        client_smape.append(mm["smape"])
        if mm.get("mase") is not None:
            client_mase.append(mm["mase"])

        rel = None
        if frozen_mse_by_client is not None:
            rel = mm["mse"] / max(frozen_mse_by_client.get(cid, 1e-8), 1e-8)
            rel_mse.append(rel)

        per_client.append(
            {
                "client_id": cid,
                "regime_id": meta_rows[cid].get("regime_id"),
                "variant_id": meta_rows[cid].get("variant_id"),
                "regime_variant": meta_rows[cid].get("regime_variant"),
                "mse": mm["mse"],
                "mae": mm["mae"],
                "smape": mm["smape"],
                "mase": mm.get("mase"),
                "val_mae": float(vv["mae"]),
                "frozen_mae": None if frozen_mae_by_client is None else frozen_mae_by_client.get(cid),
                "relative_mse": rel,
            }
        )

    return {
        "global_mse": total_se / max(total_n, 1),
        "global_mae": total_ae / max(total_n, 1),
        "global_smape": total_smape_sum / max(total_n, 1),
        "mean_client_mse": float(mean(client_mse)),
        "std_client_mse": float(pstdev(client_mse)) if len(client_mse) > 1 else 0.0,
        "mean_client_mae": float(mean(client_mae)),
        "std_client_mae": float(pstdev(client_mae)) if len(client_mae) > 1 else 0.0,
        "mean_client_smape": float(mean(client_smape)),
        "std_client_smape": float(pstdev(client_smape)) if len(client_smape) > 1 else 0.0,
        "median_client_smape": float(median(client_smape)),
        "worst_client_smape": float(max(client_smape)) if client_smape else None,
        "mean_client_mase": float(mean(client_mase)) if client_mase else None,
        "worst_client_mase": float(max(client_mase)) if client_mase else None,
        "mean_relative_mse": float(mean(rel_mse)) if rel_mse else None,
        "mean_client_val_mae": float(mean(client_val_mae)),
        "per_client": per_client,
    }


def _eval_client_val_loss(
    server,
    client: dict,
    cid: int,
    cfg: ReLoRAConfig,
    base_ctor,
    base_state: dict,
    spec,
    global_head_state: dict,
    device,
) -> float:
    """Return the validation MAE for client *cid* under the current server state."""
    with torch.no_grad():
        flat = server.generate_lora_flat([cid])[0].detach().cpu()

    model = base_ctor().to(device)
    model.load_state_dict(base_state, strict=True)
    load_forecast_head_state_dict(model, global_head_state)
    inject_lora(model, r=cfg.lora_rank, alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout, exclude_keywords=cfg.exclude_keywords)
    mark_only_lora_trainable(model)
    load_flat_lora_into_model(model, spec, flat)

    mm = evaluate_forecast(
        model,
        client["val"],
        device,
        max_batches=cfg.eval_batches,
    )
    return float(mm["mae"])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ReLoRA curriculum federated training")

    # Base config overrides (mirrors run_federated.py)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None,
                   help="Max communication rounds per curriculum round")
    p.add_argument("--local_steps", type=int, default=None)
    p.add_argument("--local_lr", type=float, default=None)
    p.add_argument("--server_lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--clients_per_round", type=int, default=None)
    p.add_argument("--eval_batches", type=int, default=None)
    p.add_argument("--train_loss", type=str, default=None, choices=["mse", "mae", "smape"])
    p.add_argument("--lora_rank", type=int, default=None)
    p.add_argument("--synthetic_regime_separability", type=str, default=None,
                   choices=["easy", "medium", "hard"])
    p.add_argument("--synthetic_config_variant", type=str, default=None)
    p.add_argument("--descriptor_set", type=str, default=None, choices=["basic", "spectral"])
    p.add_argument("--use_oracle_regime_id", action="store_true", default=None)
    p.add_argument("--force_learnable_client_embeddings", action="store_true", default=None)
    p.add_argument("--hnet_hidden", type=int, default=None)

    # ReLoRA-specific
    p.add_argument("--relora_start_regimes", type=int, default=None)
    p.add_argument("--relora_start_variants", type=int, default=None)
    p.add_argument("--relora_num_rounds", type=int, default=None)
    p.add_argument("--relora_client_patience", type=int, default=None)
    p.add_argument("--relora_adam_pruning_pct", type=float, default=None)
    p.add_argument("--relora_pruning_method", type=str, default=None,
                   choices=["magnitude", "variance", "full"])
    p.add_argument("--use-snr", dest="relora_use_snr", action="store_true", default=None,
                   help="With --relora_pruning_method variance, use SNR "
                        "(mean^2/variance) instead of raw variance as the "
                        "importance score.")
    p.add_argument("--relora_retain_fraction", type=float, default=None)
    p.add_argument("--relora_lr_warmup_steps", type=int, default=None)
    p.add_argument("--relora_lr_decay_per_round", type=float, default=None)
    p.add_argument("--relora_lr_schedule", type=str, default=None,
                   choices=["constant", "cosine"],
                   help="Within-stage server LR schedule after warmup: "
                        "'cosine' decays to relora_lr_min_factor * target_lr "
                        "by the stage's round cap.")
    p.add_argument("--relora_lr_min_factor", type=float, default=None)
    p.add_argument("--relora_patience_scope", type=str, default=None,
                   choices=["all", "sampled"],
                   help="'all' (default): every full-eval round updates every "
                        "client's patience counter. 'sampled': legacy — only "
                        "sampled clients accrue patience.")
    p.add_argument("--relora_restore_best", type=int, default=None,
                   choices=[0, 1],
                   help="1 (default): restore the stage-best (mean client val "
                        "MAE) server + head state at each stage end.")
    p.add_argument("--relora_max_rounds_per_curriculum", type=int, default=None)
    p.add_argument("--relora_max_rounds_schedule", type=str, default=None,
                   help="Explicit per-stage round-cap schedule e.g. '100,50,100,50,100,50', "
                        "parallel to relora_curriculum_schedule. Overrides "
                        "relora_max_rounds_per_curriculum.")
    p.add_argument("--relora_curriculum_schedule", type=str, default=None,
                   help="Explicit schedule e.g. '5x5,5x5,6x6,7x7'. "
                        "Overrides relora_start_regimes/variants and relora_num_rounds.")
    p.add_argument("--eval_every", type=int, default=None)

    return p


def _apply_overrides(cfg: ReLoRAConfig, args: argparse.Namespace) -> ReLoRAConfig:
    override_names = [
        "out_dir", "seed", "rounds", "local_steps", "local_lr", "server_lr",
        "batch_size", "clients_per_round", "eval_batches", "train_loss", "lora_rank",
        "synthetic_regime_separability", "synthetic_config_variant", "descriptor_set",
        "use_oracle_regime_id", "force_learnable_client_embeddings", "hnet_hidden",
        "relora_start_regimes", "relora_start_variants", "relora_num_rounds",
        "relora_client_patience", "relora_adam_pruning_pct",
        "relora_pruning_method", "relora_use_snr", "relora_retain_fraction",
        "relora_lr_warmup_steps", "relora_lr_decay_per_round",
        "relora_lr_schedule", "relora_lr_min_factor",
        "relora_patience_scope", "relora_restore_best",
        "relora_max_rounds_per_curriculum",
        "relora_max_rounds_schedule",
        "relora_curriculum_schedule",
        "eval_every",
    ]
    for name in override_names:
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    return cfg


# ---------------------------------------------------------------------------
# Per-curriculum-round training loop
# ---------------------------------------------------------------------------

def _run_one_curriculum_round(
    curriculum_round: int,
    cfg: ReLoRAConfig,
    server: Server,
    clients: list[dict],
    meta_rows: list[dict],
    base_ctor,
    base_state: dict,
    spec,
    flatdim: int,
    global_head_state: dict,
    device,
    regime_labels: torch.Tensor,
    patience_tracker: ClientPatienceTracker,
    scheduler: CurriculumScheduler,
    warmup_step_counter: list[int],  # mutable single-element list
    target_lrs: list[float],
    out_dir: str,
    csv_writer: csv.DictWriter,
    csv_file,
    frozen_mse_by_client: dict | None,
    frozen_mae_by_client: dict | None,
    max_comm_rounds: int,
) -> tuple[dict | None, dict]:
    """Run training until per-client convergence or max rounds.

    Returns:
        (final_metrics, last_row) where final_metrics is the last eval result
        and last_row is the last history CSV row written.
    """
    lora_cfg = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "exclude_keywords": cfg.exclude_keywords,
    }

    final_metrics = None
    last_row: dict = {}

    # Warn loudly when the round cap makes patience mathematically unreachable
    # (a client needs patience+1 evaluations before its counter can trigger).
    if cfg.relora_patience_scope == "sampled":
        expected_evals = max_comm_rounds * cfg.clients_per_round / max(cfg.n_clients, 1)
    else:
        expected_evals = max_comm_rounds / max(cfg.eval_every, 1)
    if expected_evals <= cfg.relora_client_patience:
        print(
            f"[relora][WARNING] stage {curriculum_round}: cap {max_comm_rounds} "
            f"allows ~{expected_evals:.0f} patience evaluations per client "
            f"(scope={cfg.relora_patience_scope}) <= patience "
            f"{cfg.relora_client_patience}; convergence-based advancement "
            f"cannot trigger — this stage will run to its round cap."
        )

    # Stage-best state (by mean client val MAE over evaluated rounds).
    best_val_mae = float("inf")
    best_snapshot: dict | None = None
    best_metrics: dict | None = None

    # Rounds consumed by LR warmup in this stage (warmup_step_counter carries
    # steps-since-boundary; stage 0 starts it at warmup_steps to skip warmup).
    warmup_rounds_this_stage = max(
        cfg.relora_lr_warmup_steps - warmup_step_counter[0], 0
    )

    for t in range(1, max_comm_rounds + 1):
        t0 = time.time()
        client_ids = server.sample_clients(cfg.clients_per_round)

        adapter_targets: dict[int, torch.Tensor] = {}
        delta_norms: list[float] = []
        local_head_states: list[dict] = []

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

            # Regression target is the updated adapter h + delta, not the
            # displacement delta: fitting delta alone has fixed point h = a*/2
            # (half the client optimum), while fitting h + delta fixes h = a*.
            adapter_targets[cid] = updated_flat
            delta_norms.append(float((updated_flat - init_flat).norm().item()))
            local_head_states.append(updated_head)

        # Hypernetwork update.
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

        # FedAvg head update.
        global_head_state = _average_state_dicts(local_head_states)

        # LR warmup, then optional within-stage cosine decay.
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

        # Legacy per-client patience: evaluate val loss for sampled clients
        # only. With relora_patience_scope="all" (default), patience is
        # instead updated for every client from the full-eval val MAEs below.
        if cfg.relora_patience_scope == "sampled":
            for cid in client_ids:
                val_loss = _eval_client_val_loss(
                    server, clients[cid], cid, cfg, base_ctor, base_state,
                    spec, global_head_state, device,
                )
                improved, converged = patience_tracker.update(cid, val_loss)
                if converged:
                    print(
                        f"[patience] client {cid} converged after "
                        f"{patience_tracker._counter.get(cid, patience_tracker.patience)} "
                        f"non-improving rounds (best val_mae={patience_tracker._best_loss[cid]:.6f})"
                    )

        secs = time.time() - t0
        do_eval = (t % cfg.eval_every) == 0 or t == 1 or t == max_comm_rounds

        if do_eval:
            metrics = _evaluate_all_clients(
                server, clients, cfg, base_ctor, base_state, spec,
                global_head_state, device, meta_rows,
                frozen_mse_by_client, frozen_mae_by_client,
            )

            # Patience from the full eval: every client's counter advances
            # every evaluated round, independent of sampling.
            if cfg.relora_patience_scope != "sampled":
                for pc in metrics["per_client"]:
                    improved, converged = patience_tracker.update(
                        pc["client_id"], pc["val_mae"]
                    )
                    if converged:
                        print(
                            f"[patience] client {pc['client_id']} converged after "
                            f"{patience_tracker._counter.get(pc['client_id'], patience_tracker.patience)} "
                            f"non-improving evals (best val_mae="
                            f"{patience_tracker._best_loss[pc['client_id']]:.6f})"
                        )

            final_metrics = dict(metrics)
            final_metrics.update(
                {
                    "curriculum_round": curriculum_round,
                    "comm_round": t,
                    "server_target_loss": float(server_losses["server_target_loss"]),
                    "server_total_loss": float(server_losses["server_total_loss"]),
                    "mean_delta_norm": float(mean(delta_norms)) if delta_norms else 0.0,
                }
            )

            # Stage-best snapshot by mean client val MAE (kept on CPU; the
            # boundary Adam reset makes the moment/weight mismatch after a
            # restore irrelevant).
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

            row = {
                "curriculum_round": curriculum_round,
                "comm_round": t,
                "server_target_loss": float(server_losses["server_target_loss"]),
                "server_total_loss": float(server_losses["server_total_loss"]),
                "mean_client_mae": float(metrics["mean_client_mae"]),
                "mean_client_mse": float(metrics["mean_client_mse"]),
                "mean_client_smape": float(metrics["mean_client_smape"]),
                "mean_client_mase": float(metrics["mean_client_mase"])
                    if metrics["mean_client_mase"] is not None else None,
                "mean_relative_mse": float(metrics["mean_relative_mse"])
                    if metrics["mean_relative_mse"] is not None else None,
                "mean_client_val_mae": float(metrics["mean_client_val_mae"]),
                "mean_delta_norm": float(mean(delta_norms)) if delta_norms else 0.0,
                "patience_converged": patience_tracker.n_converged,
                "patience_sampled": patience_tracker.n_sampled,
                "convergence_frac": patience_tracker.n_converged / max(cfg.n_clients, 1),
                "secs_round": float(secs),
                "n_clients": cfg.n_clients,
            }
            last_row = row
            csv_writer.writerow(row)
            csv_file.flush()

            print(
                f"[crnd {curriculum_round} round {t:04d}] "
                f"mae={metrics['mean_client_mae']:.5f} "
                f"smape={metrics['mean_client_smape']:.5f} "
                f"converged={patience_tracker.n_converged}/{cfg.n_clients} "
                f"secs={secs:.1f}"
            )
        else:
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
            }
            last_row = row
            csv_writer.writerow(row)
            csv_file.flush()
            print(
                f"[crnd {curriculum_round} round {t:04d}] "
                f"train_loss={float(server_losses['server_target_loss']):.5f} "
                f"converged={patience_tracker.n_converged}/{cfg.n_clients} "
                f"secs={secs:.1f}"
            )

        # Advance when every active client has exhausted its patience.
        if patience_tracker.all_converged(cfg.n_clients):
            print(
                f"[curriculum] round {curriculum_round} complete: "
                f"all {cfg.n_clients} clients exhausted patience"
            )
            break

    # Restore the stage-best server + head state so the boundary checkpoint
    # (and, for the last stage, the final checkpoint) carries the best round
    # rather than the last one. Optimizer moments still belong to the last
    # round's weights; the boundary Adam reset erases that mismatch.
    if cfg.relora_restore_best and best_snapshot is not None:
        server.hnet.load_state_dict(best_snapshot["hnet"])
        server.emb.load_state_dict(best_snapshot["emb"])
        global_head_state = best_snapshot["head"]
        if best_metrics is not None:
            final_metrics = dict(best_metrics)
            final_metrics["restored_best_round"] = best_snapshot["comm_round"]
        print(
            f"[relora] stage {curriculum_round}: restored best state from "
            f"round {best_snapshot['comm_round']} "
            f"(mean val MAE {best_val_mae:.5f}); last round was {last_row.get('comm_round')}"
        )

    return final_metrics, global_head_state, last_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _build_argparser().parse_args()
    cfg = ReLoRAConfig()
    cfg = _apply_overrides(cfg, args)

    # relora_max_rounds_per_curriculum uses cfg.rounds as fallback if not set.
    if not hasattr(cfg, "relora_max_rounds_per_curriculum") or cfg.relora_max_rounds_per_curriculum is None:
        cfg.relora_max_rounds_per_curriculum = cfg.rounds

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = _dev()
    print(f"[relora] device={device}")

    # Build base model once to get spec/flatdim.
    base_ctor = _make_base_model_ctor(cfg)
    model0 = base_ctor().to(device)
    seq_len = model0.config.seq_len

    inject_lora(
        model0,
        r=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        exclude_keywords=cfg.exclude_keywords,
    )
    mark_only_lora_trainable(model0)
    spec, flatdim = get_lora_spec_and_flatdim(model0)
    print(f"[relora] LoRA spec: {len(spec)} tensors, flatdim={flatdim:,}")

    model_base = base_ctor().to("cpu")
    base_state = model_base.state_dict()
    global_head_state = extract_forecast_head_state_dict(model_base)
    del model_base, model0

    lora_cfg = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "exclude_keywords": cfg.exclude_keywords,
    }

    scheduler = CurriculumScheduler(cfg)

    csv_path = os.path.join(cfg.out_dir, "relora_history.csv")
    csv_fieldnames = [
        "curriculum_round", "comm_round",
        "server_target_loss", "server_total_loss",
        "mean_client_mae", "mean_client_mse", "mean_client_smape",
        "mean_client_mase", "mean_relative_mse", "mean_client_val_mae",
        "mean_delta_norm",
        "patience_converged", "patience_sampled", "convergence_frac",
        "secs_round", "n_clients",
    ]

    server: Server | None = None
    old_meta_rows: list[dict] | None = None
    all_round_metrics: list[dict] = []

    # ---------------------------------------------------------------------
    # Resume: find the latest completed curriculum-round checkpoint and
    # restart from the following round (re-running that round from scratch,
    # since per-comm-round state inside a curriculum round isn't checkpointed).
    # ---------------------------------------------------------------------
    resume_k = None
    for k in range(scheduler.num_rounds()):
        ckpt_dir = os.path.join(cfg.out_dir, f"_relora_round_{k}_ckpt")
        if os.path.exists(os.path.join(ckpt_dir, "server.pt")):
            resume_k = k
    start_k = resume_k + 1 if resume_k is not None else 0

    if resume_k is not None:
        ckpt_dir = os.path.join(cfg.out_dir, f"_relora_round_{resume_k}_ckpt")
        print(f"[relora] resuming after curriculum round {resume_k} ({ckpt_dir})")

        n_regimes, n_variants = scheduler.get_round_shape(resume_k)
        cfg.synthetic_num_regimes = n_regimes
        cfg.synthetic_variants_per_regime = n_variants
        cfg.synthetic_num_clients = n_regimes * n_variants
        cfg.n_clients = cfg.synthetic_num_clients
        cfg.data_source = "synthetic"

        set_seed(cfg.seed)
        _, old_meta_rows, client_features = make_synthetic_clients(
            cfg, seq_len=seq_len, batch_size=cfg.batch_size
        )
        if getattr(cfg, "use_fixed_client_features", False):
            feat_dim = int(client_features.shape[1])
            if cfg.emb_dim != feat_dim:
                cfg.emb_dim = feat_dim

        learnable_embeddings = not cfg.use_fixed_client_features
        if cfg.force_learnable_client_embeddings:
            learnable_embeddings = True

        server = Server(
            n_clients=cfg.n_clients,
            emb_dim=cfg.emb_dim,
            hidden=cfg.hnet_hidden,
            flat_dim=flatdim,
            lr=scheduler.get_target_lr(resume_k),
            device=device,
            client_features=client_features if cfg.use_fixed_client_features else None,
            hnet_dropout=cfg.hnet_dropout,
            learnable_embeddings=learnable_embeddings,
        )
        payload = torch.load(os.path.join(ckpt_dir, "server.pt"), map_location=device)
        server.hnet.load_state_dict(payload["hnet"])
        server.emb.load_state_dict(payload["emb"])
        opt_state = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location=device)
        server.opt.load_state_dict(opt_state)
        global_head_state = torch.load(
            os.path.join(ckpt_dir, "global_head_state.pt"), map_location="cpu"
        )
        # The tracker isn't checkpointed; variance pruning at the next stage
        # boundary requires one, accumulating over the resumed stage's rounds.
        if cfg.relora_pruning_method == "variance":
            server.grad_tracker = GradVarianceTracker()

        for j in range(resume_k + 1):
            metrics_path = os.path.join(cfg.out_dir, f"round_{j}_final_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    all_round_metrics.append(json.load(f))

    csv_mode = "a" if resume_k is not None else "w"
    with open(csv_path, csv_mode, newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        if resume_k is None:
            csv_writer.writeheader()

        for k in range(start_k, scheduler.num_rounds()):
            n_regimes, n_variants = scheduler.get_round_shape(k)
            n_clients = n_regimes * n_variants
            target_lr = scheduler.get_target_lr(k)
            max_comm_rounds = scheduler.get_max_rounds(k)

            print(
                f"\n{'='*60}\n"
                f"[relora] Curriculum round {k}: "
                f"{n_regimes}x{n_variants} = {n_clients} clients, "
                f"target_lr={target_lr:.2e}, max_rounds={max_comm_rounds}\n"
                f"{'='*60}"
            )

            # -----------------------------------------------------------------
            # Build client data for this round.
            # -----------------------------------------------------------------
            cfg.synthetic_num_regimes = n_regimes
            cfg.synthetic_variants_per_regime = n_variants
            cfg.synthetic_num_clients = n_clients
            cfg.n_clients = n_clients
            cfg.data_source = "synthetic"

            set_seed(cfg.seed)  # deterministic client generation per round
            clients, meta_rows, client_features, feature_stats = make_synthetic_clients(
                cfg, seq_len=seq_len, batch_size=cfg.batch_size, return_feature_stats=True
            )
            torch.save(feature_stats, os.path.join(cfg.out_dir, "client_feature_stats.pt"))
            cfg.n_clients = len(clients)
            cfg.clients_per_round = min(cfg.clients_per_round, cfg.n_clients)

            # Normalise features before passing to server.
            if getattr(cfg, "use_fixed_client_features", False):
                feat_dim = int(client_features.shape[1])
                if cfg.emb_dim != feat_dim:
                    print(
                        f"[relora] overriding emb_dim {cfg.emb_dim} → {feat_dim} "
                        f"(fixed feature dim)"
                    )
                    cfg.emb_dim = feat_dim

            regime_labels = torch.tensor(
                [int(row["regime_id"]) for row in meta_rows], dtype=torch.long
            )

            # -----------------------------------------------------------------
            # Frozen baseline (round 0 only, reused for relative metrics).
            # -----------------------------------------------------------------
            if k == 0:
                print("[relora] computing frozen-model baseline...")
                frozen_model = base_ctor().to(device)
                frozen_mse_by_client: dict[int, float] = {}
                frozen_mae_by_client: dict[int, float] = {}
                for cid in range(cfg.n_clients):
                    mm = evaluate_forecast(
                        frozen_model, clients[cid]["test"], device,
                        max_batches=cfg.eval_batches,
                        mase_denom=clients[cid].get("mase_denom"),
                    )
                    frozen_mse_by_client[cid] = mm["mse"]
                    frozen_mae_by_client[cid] = mm["mae"]
                del frozen_model
                _save_json(
                    os.path.join(cfg.out_dir, "baseline_metrics.json"),
                    {
                        "frozen_mean_client_mse": float(mean(frozen_mse_by_client.values())),
                        "frozen_mean_client_mae": float(mean(frozen_mae_by_client.values())),
                        "n_clients": cfg.n_clients,
                    },
                )

            # -----------------------------------------------------------------
            # Server initialisation or expansion.
            # -----------------------------------------------------------------
            learnable_embeddings = not cfg.use_fixed_client_features
            if cfg.force_learnable_client_embeddings:
                learnable_embeddings = True

            if k == 0:
                server = Server(
                    n_clients=cfg.n_clients,
                    emb_dim=cfg.emb_dim,
                    hidden=cfg.hnet_hidden,
                    flat_dim=flatdim,
                    lr=target_lr,
                    device=device,
                    client_features=client_features if cfg.use_fixed_client_features else None,
                    hnet_dropout=cfg.hnet_dropout,
                    learnable_embeddings=learnable_embeddings,
                )
                if cfg.relora_pruning_method == "variance":
                    server.grad_tracker = GradVarianceTracker()
                warmup_step_counter = [cfg.relora_lr_warmup_steps]  # skip warmup in round 0
                target_lrs_for_warmup = [g["lr"] for g in server.opt.param_groups]
                print(f"[relora] server initialised ({cfg.n_clients} clients)")
            else:
                # Save end-of-round-(k-1) checkpoint.
                ckpt_dir = scheduler.save_checkpoint(server, global_head_state, cfg.out_dir, k - 1)

                # Prepare new client features (or None if fixed features not used).
                new_feats = client_features.to(device) if cfg.use_fixed_client_features else None

                # Execute boundary: Adam reset + embedding expansion + LR=0.
                # Use the *actual* previous round's shape rather than assuming
                # regimes/variants each grow by exactly 1 — curricula that hold
                # regimes fixed (or repeat a shape to force a bare Adam reset)
                # would otherwise get the wrong old_n_regimes/old_n_variants.
                prev_n_regimes, prev_n_variants = scheduler.get_round_shape(k - 1)
                scheduler.execute_boundary(
                    server=server,
                    old_meta_rows=old_meta_rows,
                    new_meta_rows=meta_rows,
                    new_client_features=new_feats,
                    old_n_regimes=prev_n_regimes,
                    old_n_variants=prev_n_variants,
                    grad_tracker=server.grad_tracker,
                )

                # Reset warmup counter so warmup fires in round k.
                warmup_step_counter = [0]
                target_lrs_for_warmup = [target_lr] * len(server.opt.param_groups)
                if cfg.relora_pruning_method == "variance":
                    pruning_desc = (
                        f"variance(retain={cfg.relora_retain_fraction:.0%}, "
                        f"use_snr={cfg.relora_use_snr})"
                    )
                elif cfg.relora_pruning_method == "full":
                    pruning_desc = "full(100%)"
                else:
                    pruning_desc = f"magnitude({cfg.relora_adam_pruning_pct:.0%})"
                print(
                    f"[relora] boundary applied: "
                    f"pruning={pruning_desc}, "
                    f"warmup={cfg.relora_lr_warmup_steps} steps, "
                    f"target_lr={target_lr:.2e}"
                )

            # Keep old meta for next boundary.
            old_meta_rows = meta_rows

            # -----------------------------------------------------------------
            # Per-client patience tracker (fresh each curriculum round).
            # -----------------------------------------------------------------
            patience_tracker = ClientPatienceTracker(
                patience=cfg.relora_client_patience,
                min_delta=cfg.relora_patience_min_delta,
            )

            # -----------------------------------------------------------------
            # Save round config.
            # -----------------------------------------------------------------
            _save_json(
                os.path.join(cfg.out_dir, f"round_{k}_config.json"),
                {
                    "curriculum_round": k,
                    "n_regimes": n_regimes,
                    "n_variants": n_variants,
                    "n_clients": cfg.n_clients,
                    "target_lr": target_lr,
                    **vars(cfg),
                },
            )
            _save_json(
                os.path.join(cfg.out_dir, f"round_{k}_client_meta.json"),
                meta_rows,
            )

            # -----------------------------------------------------------------
            # Training loop.
            # -----------------------------------------------------------------
            final_metrics, global_head_state, last_row = _run_one_curriculum_round(
                curriculum_round=k,
                cfg=cfg,
                server=server,
                clients=clients,
                meta_rows=meta_rows,
                base_ctor=base_ctor,
                base_state=base_state,
                spec=spec,
                flatdim=flatdim,
                global_head_state=global_head_state,
                device=device,
                regime_labels=regime_labels,
                patience_tracker=patience_tracker,
                scheduler=scheduler,
                warmup_step_counter=warmup_step_counter,
                target_lrs=target_lrs_for_warmup,
                out_dir=cfg.out_dir,
                csv_writer=csv_writer,
                csv_file=csv_file,
                frozen_mse_by_client=frozen_mse_by_client if k == 0 else None,
                frozen_mae_by_client=frozen_mae_by_client if k == 0 else None,
                max_comm_rounds=max_comm_rounds,
            )

            if final_metrics is not None:
                all_round_metrics.append(final_metrics)
                _save_json(
                    os.path.join(cfg.out_dir, f"round_{k}_final_metrics.json"),
                    final_metrics,
                )

    # -------------------------------------------------------------------------
    # Save final state.
    # -------------------------------------------------------------------------
    last_ckpt_dir = os.path.join(cfg.out_dir, "_relora_final_ckpt")
    os.makedirs(last_ckpt_dir, exist_ok=True)
    server.save(last_ckpt_dir)
    torch.save(global_head_state, os.path.join(last_ckpt_dir, "global_head_state.pt"))
    torch.save(base_state, os.path.join(cfg.out_dir, "base_state.pt"))
    torch.save(
        {
            "spec": spec,
            "flatdim": flatdim,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "exclude_keywords": cfg.exclude_keywords,
        },
        os.path.join(cfg.out_dir, "lora_meta.pt"),
    )
    _save_json(
        os.path.join(cfg.out_dir, "relora_all_rounds_metrics.json"),
        all_round_metrics,
    )
    _save_json(os.path.join(cfg.out_dir, "run_config.json"), vars(cfg))
    print(f"\n[relora] done — outputs in {cfg.out_dir}/")


if __name__ == "__main__":
    main()
