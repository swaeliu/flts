"""Evaluate a trained gvendi / gvendi_chronos checkpoint on the *actual*
official GIFT-eval harness -- gift_eval.data.Dataset (the real HF-backed
dataset/windowing class from /home/juliahul/projects/gift-eval) plus
gluonts.model.evaluate_model with the same metrics list used to produce
/home/juliahul/projects/gift-eval/results/chronos_bolt_small/all_results.csv
-- so the output is directly comparable to that published entry (same
prediction-length formula, same window construction, same seasonality
source, same MASE/MAE/sMAPE aggregation formula), not just similarly
shaped.

Model construction (adapter_mode=dataset: descriptor vector from each GIFT
dataset's own history -> trained hypernetwork -> LoRA adapter, zero
gradient steps on GIFT data) is delegated to whichever backbone's
build_loaded_model/load_cfg the checkpoint's own run_config.json specifies
(eval_gift.py for MOMENT, gvendi_chronos/eval_gift.py for Chronos) -- reused
unmodified, same as every other gvendi_chronos entry point in this project.

Only a single quantile ("0.5") is ever produced (see predictor.py's
docstring for why), so MSIS and mean_weighted_sum_quantile_loss are left
NaN -- everything else uses gluonts's real formula.

Usage:
    python -m gift_official_eval.run \\
        --run_dir <trained_checkpoint_dir> --results_dir <out_dir> \\
        --model_label my_10x10_model
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from gluonts.ev.metrics import MAE, MAPE, MASE, MSE, ND, NRMSE, RMSE, SMAPE
from gluonts.time_feature import get_seasonality

_ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
for _sub in ("config", "lora", "clients", "hypernet", "data_generation"):
    _p = os.path.join(_ROOT_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_EVAL_DIR = os.path.dirname(__file__)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
_GIFT_EVAL_SRC = "/home/juliahul/projects/gift-eval/src"
if _GIFT_EVAL_SRC not in sys.path:
    sys.path.insert(0, _GIFT_EVAL_SRC)

import eval_gift as _eg_moment  # noqa: E402
from gift_eval.data import Dataset as GiftEvalDataset  # noqa: E402

from .predictor import evaluate_dataset_variable_length  # noqa: E402

OUTPUT_COLUMNS = [
    "dataset", "model",
    "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]", "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]", "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS", "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain", "num_variates",
]

# Metrics computable from a single 0.5 quantile forecast (MSIS and
# MeanWeightedSumQuantileLoss need multiple quantile levels we don't have).
METRICS = [
    MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(),
    MAPE(), SMAPE(), RMSE(), NRMSE(), ND(),
]


def _select_backbone(run_dir: str) -> dict:
    """gvendi_chronos.eval_gift doesn't define its own load_cfg/build_loaded_model
    -- it monkeypatches make_base_model on the *shared* eval_gift module object
    (a sys.modules singleton). So for a Chronos checkpoint we only need to
    import it once for that side effect; every function used below still
    comes from the plain `eval_gift` module either way.
    """
    with open(os.path.join(run_dir, "run_config.json")) as f:
        d = json.load(f)
    if d.get("backbone", "moment") == "chronos":
        import gvendi_chronos.eval_gift  # noqa: F401  (side effect only)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--adapter_mode", type=str, default="dataset", choices=["dataset", "mean", "client_id"])
    p.add_argument("--client_id", type=int, default=None)
    p.add_argument("--model_label", type=str, required=True)
    p.add_argument(
        "--only_configs", type=str, default=None,
        help="Comma-separated dataset/freq/term keys to (re-)evaluate, e.g. "
             "'LOOP_SEATTLE/D/long,solar/W/long'. Appends/overwrites matching "
             "rows in an existing all_results.csv under --results_dir rather "
             "than starting over.",
    )
    args = p.parse_args()

    device = _eg_moment.dev(args.device)
    _select_backbone(args.run_dir)
    eg_module = _eg_moment
    cfg = eg_module.load_cfg(args.run_dir)
    step_horizon = int(cfg.horizon)

    configs = _eg_moment.discover_gift_configs()
    if args.only_configs:
        wanted = set(args.only_configs.split(","))
        configs = [
            rr for rr in configs
            if f"{rr['dataset_name']}/{rr['freq']}/{rr['term']}" in wanted
        ]
        print(f"Restricting to {len(configs)}/{len(wanted)} requested configs")
    else:
        print(f"Discovered {len(configs)} dataset configs")

    os.makedirs(args.results_dir, exist_ok=True)
    out_csv = os.path.join(args.results_dir, "all_results.csv")

    rows, skipped = [], []
    prior_keys = set()
    if args.only_configs and os.path.exists(out_csv):
        prior = pd.read_csv(out_csv)
        target_keys = {f"{rr['dataset_name']}/{rr['freq']}/{rr['term']}" for rr in configs}
        prior = prior[~prior["dataset"].isin(target_keys)]
        rows = prior.to_dict("records")
        prior_keys = set(prior["dataset"].astype(str))
        print(f"[merge] kept {len(prior_keys)} pre-existing rows not being re-evaluated")

    for rr in configs:
        ds_name, freq, term = rr["dataset_name"], rr["freq"], rr["term"]
        key = f"{ds_name}/{freq}/{term}"
        try:
            to_univariate = rr["num_variates"] > 1
            gdataset = GiftEvalDataset(name=f"{ds_name}/{freq}", term=term, to_univariate=to_univariate)
            season_length = get_seasonality(gdataset.freq)

            ds_hf = _eg_moment.load_gift_config(ds_name, freq)
            model = eg_module.build_loaded_model(
                cfg=cfg, run_dir=args.run_dir, device=device, ds=ds_hf,
                adapter_mode=args.adapter_mode, client_id=args.client_id,
            )
            context_len = int(model.config.seq_len)

            res = evaluate_dataset_variable_length(
                model=model, test_data=gdataset.test_data, metrics=METRICS,
                context_length=context_len, step_horizon=step_horizon, device=device,
                seasonality=season_length, mask_invalid_label=True, allow_nan_forecast=False,
            )
            row = {
                "dataset": key,
                "model": args.model_label,
                "eval_metrics/MSE[mean]": float(res["MSE[mean]"][0]),
                "eval_metrics/MSE[0.5]": float(res["MSE[0.5]"][0]),
                "eval_metrics/MAE[0.5]": float(res["MAE[0.5]"][0]),
                "eval_metrics/MASE[0.5]": float(res["MASE[0.5]"][0]),
                "eval_metrics/MAPE[0.5]": float(res["MAPE[0.5]"][0]),
                "eval_metrics/sMAPE[0.5]": float(res["sMAPE[0.5]"][0]),
                "eval_metrics/MSIS": np.nan,
                "eval_metrics/RMSE[mean]": float(res["RMSE[mean]"][0]),
                "eval_metrics/NRMSE[mean]": float(res["NRMSE[mean]"][0]),
                "eval_metrics/ND[0.5]": float(res["ND[0.5]"][0]),
                "eval_metrics/mean_weighted_sum_quantile_loss": np.nan,
                "domain": rr["domain"],
                "num_variates": rr["num_variates"],
            }
            rows.append(row)
            pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_csv, index=False)
            print(f"DONE {key}")
        except Exception as e:  # noqa: BLE001
            msg = f"SKIP {key}: {type(e).__name__}: {e}"
            skipped.append(msg)
            print(msg)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(out_csv, index=False)

    skipped_path = os.path.join(args.results_dir, "skipped.txt")
    if args.only_configs and os.path.exists(skipped_path):
        target_keys = {f"{rr['dataset_name']}/{rr['freq']}/{rr['term']}" for rr in configs}
        with open(skipped_path) as f:
            prior_skips = [
                line for line in f if not any(f"SKIP {k}:" in line for k in target_keys)
            ]
        skipped = prior_skips + [s + "\n" for s in skipped]
    with open(skipped_path, "w") as f:
        for s in skipped:
            f.write(s if s.endswith("\n") else s + "\n")
    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    main()
