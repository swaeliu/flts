
import argparse
import csv
import json
import os
import time
from statistics import mean, median, pstdev

import torch

from momentfm import MOMENTPipeline

from config import Config
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


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_base_model_ctor(cfg: Config):
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
        model = pipe
        for p in model.parameters():
            p.requires_grad = False
        return model

    return ctor


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local_steps", type=int, default=None)
    p.add_argument("--local_lr", type=float, default=None)
    p.add_argument("--server_lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--clients_per_round", type=int, default=None)
    p.add_argument("--eval_batches", type=int, default=None)
    p.add_argument("--train_loss", type=str, default=None, choices=["mse", "mae", "smape"])
    p.add_argument("--synthetic_num_clients", type=int, default=None)
    p.add_argument("--synthetic_num_regimes", type=int, default=None)
    p.add_argument("--synthetic_variants_per_regime", type=int, default=None)
    p.add_argument("--synthetic_series_per_client", type=int, default=None)
    p.add_argument("--synthetic_series_length", type=int, default=None)
    return p


def apply_overrides(cfg: Config, args):
    for name in [
        "seed",
        "rounds",
        "local_steps",
        "local_lr",
        "server_lr",
        "batch_size",
        "clients_per_round",
        "eval_batches",
        "train_loss",
        "synthetic_num_clients",
        "synthetic_num_regimes",
        "synthetic_variants_per_regime",
        "synthetic_series_per_client",
        "synthetic_series_length",
    ]:
        if hasattr(args, name):
            val = getattr(args, name)
            if val is not None:
                setattr(cfg, name, val)

    if args.tag:
        cfg.out_dir = f"{cfg.out_dir}_{args.tag}"
    return cfg


def average_state_dicts(state_dict_list):
    if not state_dict_list:
        raise ValueError("state_dict_list is empty")
    out = {}
    keys = state_dict_list[0].keys()
    for k in keys:
        vals = [sd[k].float() for sd in state_dict_list]
        out[k] = torch.stack(vals, dim=0).mean(dim=0).cpu()
    return out


def evaluate_all_clients(
    server,
    clients,
    cfg,
    base_ctor,
    base_state,
    spec,
    global_head_state,
    device,
    meta_rows,
    frozen_mse_by_client=None,
    frozen_mae_by_client=None,
    frozen_smape_by_client=None,
    frozen_mase_by_client=None,
):
    per_client = []
    client_mse, client_mae, client_smape, client_mase, rel_mse = [], [], [], [], []
    total_se, total_ae, total_smape, total_n = 0.0, 0.0, 0.0, 0

    for cid in range(cfg.n_clients):
        with torch.no_grad():
            flat = server.generate_lora_flat([cid])[0].detach().cpu()

        model = base_ctor().to(device)
        model.load_state_dict(base_state, strict=True)
        load_forecast_head_state_dict(model, global_head_state)

        inject_lora(
            model,
            r=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            exclude_keywords=cfg.exclude_keywords,
        )
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

        total_se += mm["se_sum"]
        total_ae += mm["ae_sum"]
        total_smape += mm["smape_sum"]
        total_n += mm["n"]

        client_mse.append(mm["mse"])
        client_mae.append(mm["mae"])
        client_smape.append(mm["smape"])
        if mm.get("mase") is not None:
            client_mase.append(mm["mase"])

        rel = None
        if frozen_mse_by_client is not None:
            rel = mm["mse"] / max(frozen_mse_by_client[cid], 1e-8)
            rel_mse.append(rel)

        per_client.append(
            {
                "client_id": cid,
                "dataset": meta_rows[cid].get("dataset"),
                "regime": meta_rows[cid].get("regime"),
                "regime_variant": meta_rows[cid].get("regime_variant"),
                "regime_id": meta_rows[cid].get("regime_id"),
                "variant_id": meta_rows[cid].get("variant_id"),
                "freq": meta_rows[cid].get("freq"),
                "mse": mm["mse"],
                "mae": mm["mae"],
                "smape": mm["smape"],
                "mase": mm.get("mase"),
                "median_mape": mm.get("median_mape"),
                "norm_mape": mm.get("norm_mape"),
                "n_points": mm["n"],
                "frozen_mse": None if frozen_mse_by_client is None else frozen_mse_by_client[cid],
                "frozen_mae": None if frozen_mae_by_client is None else frozen_mae_by_client[cid],
                "frozen_smape": None if frozen_smape_by_client is None else frozen_smape_by_client[cid],
                "frozen_mase": None if frozen_mase_by_client is None else frozen_mase_by_client.get(cid),
                "relative_mse": rel,
            }
        )

    return {
        "global_mse": total_se / max(total_n, 1),
        "global_mae": total_ae / max(total_n, 1),
        "global_smape": total_smape / max(total_n, 1),
        "mean_client_mse": float(mean(client_mse)),
        "std_client_mse": float(pstdev(client_mse)) if len(client_mse) > 1 else 0.0,
        "mean_client_mae": float(mean(client_mae)),
        "std_client_mae": float(pstdev(client_mae)) if len(client_mae) > 1 else 0.0,
        "mean_client_smape": float(mean(client_smape)),
        "median_client_smape": float(median(client_smape)),
        "per_client": per_client,
        "mean_client_mase": float(mean(client_mase)) if client_mase else None,
        "median_client_mase": float(median(client_mase)) if client_mase else None,
        "mean_relative_mse": float(mean(rel_mse)) if rel_mse else None,
        "median_relative_mse": float(median(rel_mse)) if rel_mse else None,
    }


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    args = build_argparser().parse_args()
    cfg = apply_overrides(Config(), args)

    cfg.data_source = "synthetic"
    if not hasattr(cfg, "synthetic_num_regimes"):
        cfg.synthetic_num_regimes = 15
    if not hasattr(cfg, "synthetic_variants_per_regime"):
        cfg.synthetic_variants_per_regime = 3

    cfg.synthetic_num_clients = cfg.synthetic_num_regimes * cfg.synthetic_variants_per_regime
    cfg.n_clients = cfg.synthetic_num_clients

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = dev()
    print("device =", device)

    base_ctor = make_base_model_ctor(cfg)

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
    print(f"[spec] #lora_tensors={len(spec)} flatdim={flatdim:,}")

    init_lora_flat_cpu = flatten_lora(model0, spec).cpu().float()

    model_base = base_ctor().to("cpu")
    base_state = model_base.state_dict()

    # Shared global forecast head, initialized from the base model
    global_head_state = extract_forecast_head_state_dict(model_base)
    print(f"[forecast head] tensors = {len(global_head_state)}")

    clients, meta_rows, client_features = make_synthetic_clients(
        cfg, seq_len=seq_len, batch_size=cfg.batch_size
    )
    cfg.n_clients = len(clients)
    cfg.clients_per_round = min(cfg.clients_per_round, cfg.n_clients)

    if getattr(cfg, "use_fixed_client_features", False):
        feat_dim = int(client_features.shape[1])
        if cfg.emb_dim != feat_dim:
            print(
                f"[info] overriding cfg.emb_dim from {cfg.emb_dim} "
                f"to fixed client feature dim {feat_dim}"
            )
            cfg.emb_dim = feat_dim

    print(f"[client construction] total synthetic clients = {cfg.n_clients}")
    print(
        f"[client construction] regimes = {cfg.synthetic_num_regimes}, "
        f"variants/regime = {cfg.synthetic_variants_per_regime}"
    )

    server = Server(
        n_clients=cfg.n_clients,
        emb_dim=cfg.emb_dim,
        hidden=cfg.hnet_hidden,
        flat_dim=flatdim,
        lr=cfg.server_lr,
        device=device,
        client_features=client_features if cfg.use_fixed_client_features else None,
        hnet_dropout=cfg.hnet_dropout,
        learnable_embeddings=not cfg.use_fixed_client_features,
    )

    lora_cfg = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "exclude_keywords": cfg.exclude_keywords,
    }

    frozen_model = base_ctor().to(device)
    frozen_mse_by_client = {}
    frozen_mae_by_client = {}
    frozen_smape_by_client = {}
    frozen_mase_by_client = {}

    for cid in range(cfg.n_clients):
        mm = evaluate_forecast(
            frozen_model,
            clients[cid]["test"],
            device,
            max_batches=cfg.eval_batches,
            mase_denom=clients[cid].get("mase_denom"),
            seasonal_naive_mape=clients[cid].get("seasonal_naive_mape"),
        )
        frozen_mse_by_client[cid] = mm["mse"]
        frozen_mae_by_client[cid] = mm["mae"]
        frozen_smape_by_client[cid] = mm["smape"]
        if mm.get("mase") is not None:
            frozen_mase_by_client[cid] = mm["mase"]

    save_json(os.path.join(cfg.out_dir, "synthetic_client_meta.json"), meta_rows)
    save_json(os.path.join(cfg.out_dir, "run_config.json"), vars(cfg))
    save_json(
        os.path.join(cfg.out_dir, "baseline_metrics.json"),
        {
            "frozen_mean_client_mse": float(mean(frozen_mse_by_client.values())),
            "frozen_mean_client_mae": float(mean(frozen_mae_by_client.values())),
            "frozen_mean_client_smape": float(mean(frozen_smape_by_client.values())),
            "frozen_mean_client_mase": float(mean(frozen_mase_by_client.values()))
            if frozen_mase_by_client
            else None,
        },
    )

    csv_path = os.path.join(cfg.out_dir, "history.csv")
    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(
            csv_f,
            fieldnames=[
                "round",
                "server_target_loss",
                "global_test_mse",
                "global_test_mae",
                "global_test_smape",
                "mean_client_mse",
                "mean_client_mae",
                "mean_client_smape",
                "mean_client_mase",
                "mean_relative_mse",
                "mean_delta_norm",
                "secs_round",
            ],
        )
        writer.writeheader()

        final_metrics = None

        for t in range(1, cfg.rounds + 1):
            t0 = time.time()
            client_ids = server.sample_clients(cfg.clients_per_round)

            delta_targets = {}
            delta_norms = []
            local_head_states = []

            for cid in client_ids:
                init_flat = init_lora_flat_cpu.clone()

                updated_flat, updated_head_state = local_train_lora_and_head_steps(
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

                delta = updated_flat - init_flat
                delta_targets[cid] = delta
                delta_norms.append(float(delta.norm().item()))
                local_head_states.append(updated_head_state)

            # Hypernetwork update for client-specific LoRA
            server_target_loss = server.update_from_deltas(client_ids, delta_targets)

            # FedAvg update for shared forecast head
            global_head_state = average_state_dicts(local_head_states)

            if (t % cfg.eval_every) == 0 or t == 1 or t == cfg.rounds:
                metrics = evaluate_all_clients(
                    server,
                    clients,
                    cfg,
                    base_ctor,
                    base_state,
                    spec,
                    global_head_state,
                    device,
                    meta_rows,
                    frozen_mse_by_client,
                    frozen_mae_by_client,
                    frozen_smape_by_client,
                    frozen_mase_by_client,
                )
                final_metrics = metrics
                secs_round = time.time() - t0

                row = {
                    "round": t,
                    "server_target_loss": float(server_target_loss),
                    "global_test_mse": float(metrics["global_mse"]),
                    "global_test_mae": float(metrics["global_mae"]),
                    "global_test_smape": float(metrics["global_smape"]),
                    "mean_client_mse": float(metrics["mean_client_mse"]),
                    "mean_client_mae": float(metrics["mean_client_mae"]),
                    "mean_client_smape": float(metrics["mean_client_smape"]),
                    "mean_client_mase": float(metrics["mean_client_mase"])
                    if metrics["mean_client_mase"] is not None
                    else None,
                    "mean_relative_mse": float(metrics["mean_relative_mse"])
                    if metrics["mean_relative_mse"] is not None
                    else None,
                    "mean_delta_norm": float(mean(delta_norms)) if delta_norms else 0.0,
                    "secs_round": float(secs_round),
                }
                writer.writerow(row)
                csv_f.flush()

                print(
                    f"[round {t:03d}] "
                    f"delta_loss={server_target_loss:.6f} "
                    f"global_mse={metrics['global_mse']:.6f} "
                    f"global_mae={metrics['global_mae']:.6f} "
                    f"global_smape={metrics['global_smape']:.6f} "
                    f"mean_client_mase={metrics['mean_client_mase']} "
                    f"rel_mse={metrics['mean_relative_mse']} "
                    f"mean_delta_norm={row['mean_delta_norm']:.6f} "
                    f"secs={secs_round:.2f}"
                )
            else:
                secs_round = time.time() - t0
                print(
                    f"[round {t:03d}] "
                    f"train-only delta_loss={float(server_target_loss):.6f} "
                    f"mean_delta_norm={float(mean(delta_norms)) if delta_norms else 0.0:.6f} "
                    f"secs={secs_round:.2f}"
                )

    if final_metrics is not None:
        save_json(os.path.join(cfg.out_dir, "final_metrics.json"), final_metrics)
        with open(os.path.join(cfg.out_dir, "final_per_client_metrics.csv"), "w", newline="") as f:
            fieldnames = list(final_metrics["per_client"][0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in final_metrics["per_client"]:
                w.writerow(row)

    torch.save(global_head_state, os.path.join(cfg.out_dir, "global_head_state.pt"))

    save_json(
        os.path.join(cfg.out_dir, "final_global_head_state_keys.json"),
        list(global_head_state.keys()),
    )

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

    server.save(cfg.out_dir)
    print(f"Saved clustered synthetic experiment to: {cfg.out_dir}/")


if __name__ == "__main__":
    main()