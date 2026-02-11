#!/usr/bin/env python3
import os
import json
import torch
import time
import csv
from statistics import mean, pstdev

from momentfm import MOMENTPipeline

from config import Config
from data import set_seed, make_series, make_client_loaders
from lora_utils import (
    inject_lora,
    mark_only_lora_trainable,
    get_lora_spec_and_flatdim,
)
from client import local_train_lora_steps, evaluate_forecast
from server import Server
from lora_utils import load_flat_lora_into_model


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


def main():
    cfg = Config()
    set_seed(0)
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


    model_base = base_ctor().to("cpu")
    base_state = model_base.state_dict()


    clients = []
    for i in range(cfg.n_clients):
        series = make_series(cfg.total_steps, cfg.n_channels, seed=0, client_id=i)
        train_loader, val_loader, test_loader = make_client_loaders(
            series=series,
            seq_len=seq_len,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
        )
        clients.append({"train": train_loader, "val": val_loader, "test": test_loader})

    server = Server(
        n_clients=cfg.n_clients,
        emb_dim=cfg.emb_dim,
        hidden=cfg.hnet_hidden,
        flat_dim=flatdim,
        lr=cfg.server_lr,
        device=device,
    )

    lora_cfg = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "exclude_keywords": cfg.exclude_keywords,
    }

    frozen_model = base_ctor().to(device)
    base_metrics = evaluate_forecast(frozen_model, clients[0]["test"], device)
    print(f"[baseline client0] mse={base_metrics['mse']:.6f} mae={base_metrics['mae']:.6f}")


    os.makedirs(cfg.out_dir, exist_ok=True)
    csv_path = os.path.join(cfg.out_dir, "history.csv")
    csv_exists = os.path.exists(csv_path)

    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        "round",
        "delta_alignment",
        "client0_test_mse",
        "client0_test_mae",
        "mean_test_mse",
        "std_test_mse",
        "mean_test_mae",
        "std_test_mae",
        "mean_delta_norm",
        "std_delta_norm",
        "clients_sampled",
        "secs_round",
    ])
    if not csv_exists:
        csv_w.writeheader()
        csv_f.flush()

    history = []


    for t in range(1, cfg.rounds + 1):
        round_t0 = time.time()

        client_ids = server.sample_clients(cfg.clients_per_round)
        pred_flats = server.generate_lora_flat(client_ids).detach().cpu() 

     
        deltas = {}
        delta_norms = []
        for j, cid in enumerate(client_ids):
            init_flat = pred_flats[j].contiguous()  # CPU

            updated_flat = local_train_lora_steps(
                base_model_ctor=base_ctor,
                base_state_dict=base_state,
                spec=spec,
                init_lora_flat=init_flat,
                train_loader=clients[cid]["train"],
                device=device,
                local_steps=cfg.local_steps,
                lr=cfg.local_lr,
                lora_cfg=lora_cfg,
            )

            d = (updated_flat - init_flat)  # CPU
            deltas[cid] = d
            delta_norms.append(float(d.norm().item()))

      
        alignment = server.update_from_deltas(client_ids, deltas)

 
        test_mse_all = []
        test_mae_all = []
        for cid_eval in range(cfg.n_clients):
            with torch.no_grad():
                flat = server.generate_lora_flat([cid_eval])[0].detach().cpu()

            eval_model = base_ctor().to(device)
            eval_model.load_state_dict(base_state, strict=True)
            inject_lora(eval_model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, cfg.exclude_keywords)
            mark_only_lora_trainable(eval_model)
            load_flat_lora_into_model(eval_model, spec, flat)

            mm = evaluate_forecast(eval_model, clients[cid_eval]["test"], device)
            test_mse_all.append(mm["mse"])
            test_mae_all.append(mm["mae"])

        client0_mse = float(test_mse_all[0])
        client0_mae = float(test_mae_all[0])

        mean_mse = float(mean(test_mse_all))
        std_mse = float(pstdev(test_mse_all)) if len(test_mse_all) > 1 else 0.0
        mean_mae = float(mean(test_mae_all))
        std_mae = float(pstdev(test_mae_all)) if len(test_mae_all) > 1 else 0.0

        mean_dn = float(mean(delta_norms)) if len(delta_norms) else 0.0
        std_dn = float(pstdev(delta_norms)) if len(delta_norms) > 1 else 0.0

        secs_round = time.time() - round_t0

        print(
            f"[round {t:03d}] delta_alignment={alignment:.6e} "
            f"client0_test_mse={client0_mse:.6f} mae={client0_mae:.6f} "
            f"mean_test_mse={mean_mse:.6f}±{std_mse:.6f} "
            f"mean_delta_norm={mean_dn:.3f}"
        )

        row = {
            "round": t,
            "delta_alignment": float(alignment),
            "client0_test_mse": client0_mse,
            "client0_test_mae": client0_mae,
            "mean_test_mse": mean_mse,
            "std_test_mse": std_mse,
            "mean_test_mae": mean_mae,
            "std_test_mae": std_mae,
            "mean_delta_norm": mean_dn,
            "std_delta_norm": std_dn,
            "clients_sampled": ",".join(map(str, client_ids)),
            "secs_round": float(secs_round),
        }
        history.append(row)
        csv_w.writerow(row)
        csv_f.flush()


    server.save(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    csv_f.close()
    print(f"Saved server checkpoint + history to: {cfg.out_dir}/")


if __name__ == "__main__":
    main()
