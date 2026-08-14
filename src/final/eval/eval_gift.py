import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from momentfm import MOMENTPipeline

from config import Config
from server import Server
from client import load_forecast_head_state_dict
from lora_utils import (
    inject_lora,
    load_flat_lora_into_model,
    mark_only_lora_trainable,
)
from synthetic_data import estimate_client_feature_vector_from_series_list


GIFT_ROOT = "/home/juliahul/projects/flts/data/datasets/gift_eval"

OUTPUT_COLUMNS = [
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
]


def dev(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_cfg(run_dir: str) -> Config:
    cfg = Config()
    with open(os.path.join(run_dir, "run_config.json"), "r") as f:
        cfg_dict = json.load(f)
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def make_base_model(cfg: Config):
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


def reconstruct_server(cfg: Config, lora_meta: dict, server_payload: dict, device: torch.device) -> Server:
    emb_weight = server_payload["emb"]["weight"]
    n_clients = emb_weight.shape[0]

    client_features = server_payload.get("client_features", None)
    learnable_embeddings = server_payload.get("learnable_embeddings", True)

    server = Server(
        n_clients=n_clients,
        emb_dim=cfg.emb_dim,
        hidden=cfg.hnet_hidden,
        flat_dim=lora_meta["flatdim"],
        lr=cfg.server_lr,
        device=device,
        client_features=client_features,
        hnet_dropout=getattr(cfg, "hnet_dropout", 0.10),
        learnable_embeddings=learnable_embeddings,
    )
    server.hnet.load_state_dict(server_payload["hnet"], strict=True)
    server.emb.load_state_dict(server_payload["emb"], strict=True)
    server.hnet.eval()
    server.emb.eval()
    return server


@torch.no_grad()
def build_adapter_flat(server: Server, adapter_mode: str, client_id: int | None = None) -> torch.Tensor:
    n_clients = server.emb.num_embeddings

    if adapter_mode == "mean":
        client_ids = list(range(n_clients))
        flat = server.generate_lora_flat(client_ids).mean(dim=0)
        return flat.detach().cpu()

    if adapter_mode == "client_id":
        if client_id is None:
            raise ValueError("--client_id is required when --adapter_mode client_id")
        if client_id < 0 or client_id >= n_clients:
            raise ValueError(f"client_id {client_id} out of range [0, {n_clients - 1}]")
        flat = server.generate_lora_flat([client_id])[0]
        return flat.detach().cpu()

    raise ValueError(f"Unknown adapter_mode: {adapter_mode}")


def gift_config_path(dataset_name: str, freq: str) -> str:
    return os.path.join(GIFT_ROOT, dataset_name, freq)


def load_gift_config(dataset_name: str, freq: str):
    path = gift_config_path(dataset_name, freq)
    if not os.path.exists(path):
        raise FileNotFoundError(f"GIFT dataset path does not exist: {path}")
    return load_from_disk(path)


def get_example_for_num_variates(ds):
    if isinstance(ds, DatasetDict):
        if "train" in ds and len(ds["train"]) > 0:
            return ds["train"][0]
        for k in ds.keys():
            if len(ds[k]) > 0:
                return ds[k][0]
        return None

    if isinstance(ds, Dataset):
        if len(ds) == 0:
            return None
        return ds[0]

    return None


def split_term_name(term: str) -> str:
    t = term.lower()
    if t not in {"short", "medium", "long"}:
        raise ValueError(f"Unexpected term: {term}")
    return t


def prettify_dataset_name(name: str) -> str:
    return name


def infer_domain(dataset_name: str) -> str:
    lower = dataset_name.lower()

    if any(x in lower for x in ["electric", "ett", "solar"]):
        return "Energy"
    if any(x in lower for x in ["birth", "covid", "hospital"]):
        return "Healthcare"
    if any(x in lower for x in ["taxi", "traffic", "transport"]):
        return "Transport"
    if any(x in lower for x in ["restaurant", "sales", "bizitobs", "bitbrains", "cloud"]):
        return "Web/CloudOps"
    if any(x in lower for x in ["weather", "temperature", "rain", "saugeen"]):
        return "Nature"
    if any(x in lower for x in ["m4", "car_parts", "hierarchical"]):
        return "Industry"
    return dataset_name


def get_first_available_array(item: dict):
    for key in ["input", "target", "history", "past_values", "x"]:
        if key in item:
            return np.asarray(item[key], dtype=np.float32)
    raise KeyError(f"Could not find an input/target/history array in item keys: {list(item.keys())}")


def get_num_variates_from_array(arr: np.ndarray) -> float:
    if arr.ndim == 1:
        return 1.0
    return float(arr.shape[0])


def infer_num_variates(dataset_name: str, freq: str) -> float:
    ds = load_gift_config(dataset_name, freq)
    item = get_example_for_num_variates(ds)
    if item is None:
        return np.nan
    arr = get_first_available_array(item)
    return get_num_variates_from_array(arr)


def discover_gift_configs():
    configs = []
    terms = ["short", "medium", "long"]

    if not os.path.exists(GIFT_ROOT):
        raise FileNotFoundError(f"GIFT_ROOT does not exist: {GIFT_ROOT}")

    for dataset_name in sorted(os.listdir(GIFT_ROOT)):
        dataset_path = os.path.join(GIFT_ROOT, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        if dataset_name in {"downloads", "hub"} or dataset_name.startswith("_"):
            continue

        found_any_freq = False

        for freq in sorted(os.listdir(dataset_path)):
            freq_path = os.path.join(dataset_path, freq)
            if not os.path.isdir(freq_path):
                continue

            if not os.path.exists(os.path.join(freq_path, "dataset_info.json")):
                continue

            found_any_freq = True
            pretty_name = prettify_dataset_name(dataset_name)
            domain = infer_domain(dataset_name)
            num_variates = infer_num_variates(dataset_name, freq)

            for term in terms:
                configs.append(
                    {
                        "dataset_name": pretty_name,
                        "freq": freq,
                        "term": term,
                        "domain": domain,
                        "num_variates": num_variates,
                    }
                )

        if not found_any_freq:
            continue

    return configs


def flatten_valid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def paired_valid(pred: np.ndarray, true: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(true)
    return pred[mask], true[mask]


def compute_mase_scale_from_train_ds(train_ds) -> float:
    denom_sum = 0.0
    denom_n = 0

    for item in train_ds:
        hist = get_first_available_array(item)
        if hist.ndim == 1:
            hist = hist[None, :]

        diffs = np.abs(hist[:, 1:] - hist[:, :-1])
        if diffs.size > 0:
            valid = diffs[np.isfinite(diffs)]
            denom_sum += float(valid.sum())
            denom_n += int(valid.size)

    if denom_n == 0:
        return np.nan
    return denom_sum / max(denom_n, 1)


@torch.no_grad()
def forecast_one_item(model, past: np.ndarray, device: torch.device) -> np.ndarray:
    if past.ndim == 1:
        past = past[None, :]

    x = torch.from_numpy(past[None, :, :].astype(np.float32)).to(device)  # [1, C, L]
    out = model(x_enc=x)
    pred = out.forecast if hasattr(out, "forecast") else out
    pred = pred.detach().cpu().float().numpy()

    if pred.ndim == 1:
        pred = pred[None, :]
    elif pred.ndim == 3:
        pred = pred[0]

    return pred


def get_term_horizon_from_series_length(term: str, total_len: int, max_horizon: int) -> int:
    t = split_term_name(term)

    if t == "short":
        h = max(1, max_horizon // 4)
    elif t == "medium":
        h = max(1, max_horizon // 2)
    else:
        h = max_horizon

    h = min(h, max(1, total_len - 1))
    return h


# ---------------------------------------------------------------------------
# Official GIFT-eval protocol (gift_eval/data.py): prediction length is a
# per-base-frequency constant times a term multiplier, NOT capped by the
# model's native horizon; test uses rolling windows with stride pred_len;
# MASE is scaled by the *seasonal* naive error.
# ---------------------------------------------------------------------------

GIFT_PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
GIFT_TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}
# gluonts default seasonalities, divided by the frequency multiple
GIFT_SEASONALITY_MAP = {"S": 3600, "T": 1440, "H": 24, "D": 1, "W": 1, "M": 12, "Q": 4, "A": 1}
GIFT_TEST_SPLIT = 0.1
GIFT_MAX_WINDOW = 20


def parse_freq(freq: str) -> tuple[int, str]:
    m = re.match(r"^(\d*)\s*([A-Za-z]+)$", freq.strip())
    if m is None:
        raise ValueError(f"cannot parse freq {freq!r}")
    mult = int(m.group(1)) if m.group(1) else 1
    base = {"MIN": "T"}.get(m.group(2).upper(), m.group(2).upper())
    return mult, base


def official_pred_len(freq: str, term: str) -> int:
    _, base = parse_freq(freq)
    return GIFT_PRED_LENGTH_MAP[base] * GIFT_TERM_MULTIPLIER[term.lower()]


def official_seasonality(freq: str) -> int:
    mult, base = parse_freq(freq)
    return max(1, GIFT_SEASONALITY_MAP.get(base, 1) // mult)


@torch.no_grad()
def forecast_rollout_batched(
    model,
    ctxs: np.ndarray,
    pred_len: int,
    device: torch.device,
    step_horizon: int,
    chunk: int = 128,
) -> np.ndarray:
    """Autoregressive rollout to pred_len for a batch of contexts [B, L].

    Rows are independent series/windows; MOMENT is channel-independent, so a
    chunk of rows is fed as channels of one forward pass.
    """
    B, L = ctxs.shape
    out = np.empty((B, pred_len), dtype=np.float32)
    n_steps = int(math.ceil(pred_len / step_horizon))

    for s0 in range(0, B, chunk):
        ctx = ctxs[s0 : s0 + chunk].astype(np.float32).copy()
        preds = []
        for _ in range(n_steps):
            x = torch.from_numpy(ctx[None, :, :]).to(device)
            o = model(x_enc=x)
            p = (o.forecast if hasattr(o, "forecast") else o).detach().cpu().float().numpy()
            if p.ndim == 3:
                p = p[0]
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            preds.append(p)
            ctx = np.concatenate([ctx, p], axis=1)[:, -L:]
        out[s0 : s0 + chunk] = np.concatenate(preds, axis=1)[:, :pred_len]

    return out


def load_feature_norm_stats(run_dir: str, server_payload: dict):
    """
    Prefer saved normalization stats if present; otherwise fall back to
    recomputing them from the saved synthetic client_features.
    """
    stats_path = os.path.join(run_dir, "client_feature_stats.pt")
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location="cpu")
        mu = stats["mean"].float().view(1, -1)
        sd = stats["std"].float().view(1, -1).clamp_min(1e-6)
        return mu, sd

    client_features = server_payload.get("client_features", None)
    if client_features is None:
        raise FileNotFoundError(
            f"Missing {stats_path}, and server.pt does not contain client_features for fallback normalization."
        )

    feats = client_features.detach().float().cpu()
    mu = feats.mean(dim=0, keepdim=True)
    sd = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mu, sd


def collect_series_list_from_gift_dataset(ds):
    """
    Treat each univariate channel of each series as an observed series for
    client-feature estimation.
    """
    series_list: List[np.ndarray] = []

    for item in ds:
        if "target" not in item:
            continue
        target = np.asarray(item["target"], dtype=np.float32)

        if target.ndim == 1:
            series_list.append(target)
        else:
            for c in range(target.shape[0]):
                series_list.append(target[c])

    return series_list


@torch.no_grad()
def build_dataset_adapter_flat(
    cfg: Config,
    run_dir: str,
    device: torch.device,
    ds,
) -> torch.Tensor:
    """
    Estimate a dataset/client feature vector from observed GIFT histories and
    feed it directly through the trained hypernetwork.
    """
    lora_meta = torch.load(os.path.join(run_dir, "lora_meta.pt"), map_location="cpu")
    server_payload = torch.load(os.path.join(run_dir, "server.pt"), map_location="cpu")
    server = reconstruct_server(cfg, lora_meta, server_payload, device)

    if getattr(server, "learnable_embeddings", False):
        raise RuntimeError(
            "adapter_mode='dataset' expects fixed feature conditioning. "
            "This checkpoint appears to use learnable embeddings."
        )

    seq_len = int(make_base_model(cfg).config.seq_len)
    series_list = collect_series_list_from_gift_dataset(ds)
    feat = estimate_client_feature_vector_from_series_list(
        series_list,
        seq_len=seq_len,
        max_windows_per_series=int(getattr(cfg, "client_feature_windows_per_series", 8)),
    ).view(1, -1)

    mu, sd = load_feature_norm_stats(run_dir, server_payload)
    feat = (feat - mu) / sd
    feat = feat.to(device)

    flat = server.hnet(feat)[0]
    return flat.detach().cpu()


@torch.no_grad()
def build_loaded_model(
    cfg: Config,
    run_dir: str,
    device: torch.device,
    ds,
    adapter_mode: str,
    client_id: int | None,
):
    _base_path = os.path.join(run_dir, "base_state.pt")
    if not os.path.exists(_base_path):
        # base_state.pt is identical across all runs; fall back to the shared copy
        _base_path = os.path.join(
            os.path.dirname(os.path.realpath(run_dir)), "base_state.pt"
        )
    base_state = torch.load(_base_path, map_location="cpu")
    lora_meta = torch.load(os.path.join(run_dir, "lora_meta.pt"), map_location="cpu")
    server_payload = torch.load(os.path.join(run_dir, "server.pt"), map_location="cpu")
    global_head_path = os.path.join(run_dir, "global_head_state.pt")
    if not os.path.exists(global_head_path):
        raise FileNotFoundError(
            f"Missing {global_head_path}. Re-run training after saving the final forecast head."
        )
    global_head_state = torch.load(global_head_path, map_location="cpu")

    server = reconstruct_server(cfg, lora_meta, server_payload, device)

    if adapter_mode == "dataset":
        flat = build_dataset_adapter_flat(
            cfg=cfg,
            run_dir=run_dir,
            device=device,
            ds=ds,
        )
    else:
        flat = build_adapter_flat(server, adapter_mode=adapter_mode, client_id=client_id)

    model = make_base_model(cfg).to(device)
    model.load_state_dict(base_state, strict=True)
    load_forecast_head_state_dict(model, global_head_state)

    inject_lora(
        model,
        r=lora_meta["lora_rank"],
        alpha=lora_meta["lora_alpha"],
        dropout=lora_meta["lora_dropout"],
        exclude_keywords=tuple(lora_meta["exclude_keywords"]),
    )
    mark_only_lora_trainable(model)
    load_flat_lora_into_model(model, lora_meta["spec"], flat)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def evaluate_one_config(
    cfg: Config,
    run_dir: str,
    device: torch.device,
    dataset_name: str,
    freq: str,
    term: str,
    domain: str,
    num_variates: float,
    adapter_mode: str,
    client_id: int | None,
):
    ds = load_gift_config(dataset_name, freq)

    if len(ds) == 0:
        return None, f"SKIP {dataset_name}/{freq}/{term}: empty dataset"

    model = build_loaded_model(
        cfg=cfg,
        run_dir=run_dir,
        device=device,
        ds=ds,
        adapter_mode=adapter_mode,
        client_id=client_id,
    )

    context_len = int(model.config.seq_len)
    step_horizon = int(cfg.horizon)

    pred_len = official_pred_len(freq, term)
    seasonality = official_seasonality(freq)

    # Pass 1: series lengths -> official rolling-window count (from the
    # shortest series, stride pred_len, capped at GIFT_MAX_WINDOW).
    lengths = []
    for item in ds:
        if "target" not in item:
            return None, f"SKIP {dataset_name}/{freq}/{term}: missing target column"
        t = np.asarray(item["target"], dtype=np.float32)
        if t.ndim == 1:
            t = t[None, :]
        lengths.append(t.shape[1])

    if not lengths:
        return None, f"SKIP {dataset_name}/{freq}/{term}: empty dataset"

    min_T = min(lengths)
    if min_T <= pred_len:
        return None, (
            f"SKIP {dataset_name}/{freq}/{term}: shortest series ({min_T}) "
            f"cannot fit one horizon-{pred_len} test window"
        )
    n_windows = int(min(max(1, math.ceil(GIFT_TEST_SPLIT * min_T / pred_len)), GIFT_MAX_WINDOW))

    # Pass 2: collect all (series x variate x window) contexts and targets,
    # and accumulate the seasonal-naive MASE denominator over each window's
    # history.
    mase_denom_sum = 0.0
    mase_denom_n = 0
    ctx_rows, fut_rows = [], []

    for item in ds:
        target = np.asarray(item["target"], dtype=np.float32)
        if target.ndim == 1:
            target = target[None, :]
        C, T = target.shape

        for w in range(1, n_windows + 1):
            hist_end = T - w * pred_len
            if hist_end < 1:
                continue
            past = target[:, :hist_end]
            future = target[:, hist_end : hist_end + pred_len]

            ctx = np.nan_to_num(past[:, -context_len:], nan=0.0, posinf=0.0, neginf=0.0)
            if ctx.shape[1] < context_len:
                pad = np.zeros((C, context_len - ctx.shape[1]), dtype=np.float32)
                ctx = np.concatenate([pad, ctx], axis=1)
            ctx_rows.append(ctx)
            fut_rows.append(future)

            m = min(seasonality, max(1, past.shape[1] - 1))
            diffs = np.abs(past[:, m:] - past[:, :-m])
            valid_diffs = diffs[np.isfinite(diffs)]
            if valid_diffs.size > 0:
                mase_denom_sum += float(valid_diffs.sum())
                mase_denom_n += int(valid_diffs.size)

    if not ctx_rows:
        return None, f"SKIP {dataset_name}/{freq}/{term}: no valid test windows"

    ctxs = np.concatenate(ctx_rows, axis=0)
    futs = np.concatenate(fut_rows, axis=0)
    preds = forecast_rollout_batched(model, ctxs, pred_len, device, step_horizon)

    pred_flat, true_flat = paired_valid(preds, futs)
    if true_flat.size == 0:
        return None, f"SKIP {dataset_name}/{freq}/{term}: no valid finite targets"

    err = pred_flat - true_flat
    n_total = int(true_flat.size)
    se_sum = float(np.sum(err ** 2))
    ae_sum = float(np.sum(np.abs(err)))
    smape_sum = float(
        np.sum(2.0 * np.abs(err) / np.maximum(np.abs(pred_flat) + np.abs(true_flat), 1e-8))
    )
    ape_num = float(np.sum(np.abs(err) / np.maximum(np.abs(true_flat), 1e-8)))
    ape_den = n_total
    true_abs_sum = float(np.sum(np.abs(true_flat)))
    true_all = [true_flat]

    mse = se_sum / n_total
    mae = ae_sum / n_total
    rmse = float(np.sqrt(mse))
    smape = smape_sum / n_total
    mape = ape_num / max(ape_den, 1)

    nd = ae_sum / max(true_abs_sum, 1e-8)

    if true_all:
        true_concat = np.concatenate(true_all)
        scale = max(float(np.mean(np.abs(true_concat))), 1e-8)
        nrmse = rmse / scale
    else:
        nrmse = np.nan

    if mase_denom_n > 0:
        mase_scale = mase_denom_sum / mase_denom_n
        mase = mae / max(mase_scale, 1e-8)
    else:
        mase = np.nan

    row = {
        "dataset": f"{dataset_name}/{freq}/{term}",
        "model": Path(run_dir).name,
        "eval_metrics/MSE[mean]": float(mse),
        "eval_metrics/MSE[0.5]": float(mse),
        "eval_metrics/MAE[0.5]": float(mae),
        "eval_metrics/MASE[0.5]": float(mase) if np.isfinite(mase) else np.nan,
        "eval_metrics/MAPE[0.5]": float(mape),
        "eval_metrics/sMAPE[0.5]": float(smape),
        "eval_metrics/MSIS": np.nan,
        "eval_metrics/RMSE[mean]": float(rmse),
        "eval_metrics/NRMSE[mean]": float(nrmse) if np.isfinite(nrmse) else np.nan,
        "eval_metrics/ND[0.5]": float(nd),
        "eval_metrics/mean_weighted_sum_quantile_loss": np.nan,
        "domain": domain,
        "num_variates": float(num_variates),
    }

    return row, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--adapter_mode",
        type=str,
        default="dataset",
        choices=["dataset", "mean", "client_id"],
    )
    p.add_argument("--client_id", type=int, default=None)
    args = p.parse_args()

    device = dev(args.device)
    cfg = load_cfg(args.run_dir)

    configs = discover_gift_configs()
    print(f"Discovered {len(configs)} dataset configs from {GIFT_ROOT}")

    rows = []
    skipped = []

    os.makedirs(args.results_dir, exist_ok=True)
    out_csv = os.path.join(args.results_dir, "all_results.csv")

    # Resume: skip configs already present in an earlier partial csv.
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(prev["dataset"].astype(str))
        print(f"[resume] found {len(done)} already-evaluated configs in {out_csv}")

    for rr in configs:
        key = f"{rr['dataset_name']}/{rr['freq']}/{rr['term']}"
        if key in done:
            continue
        row, msg = evaluate_one_config(
            cfg=cfg,
            run_dir=args.run_dir,
            device=device,
            dataset_name=rr["dataset_name"],
            freq=rr["freq"],
            term=rr["term"],
            domain=rr["domain"],
            num_variates=rr["num_variates"],
            adapter_mode=args.adapter_mode,
            client_id=args.client_id,
        )

        if row is None:
            skipped.append(msg)
            print(msg)
        else:
            rows.append(row)
            # Write incrementally so a walltime kill loses nothing.
            pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_csv, index=False)
            print(f"DONE {row['dataset']}")

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(out_csv, index=False)

    skipped_path = os.path.join(args.results_dir, "skipped.txt")
    with open(skipped_path, "w") as f:
        for s in skipped:
            f.write(str(s) + "\n")

    print(f"\nSaved results to: {out_csv}")
    print(f"Saved skipped configs to: {skipped_path}")


if __name__ == "__main__":
    main()