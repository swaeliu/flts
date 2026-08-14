"""Builds per-item forecasts for a real gift_eval.data.Dataset test split from
an already-adapted gvendi/gvendi_chronos model (backbone + trained LoRA
adapter + trained forecast head, as produced by eval_gift.py's or
gvendi_chronos/eval_gift.py's build_loaded_model).

This does NOT go through gluonts's Predictor.predict(dataset) protocol,
because that assumes one fixed prediction_length per predictor instance --
true for the nominal per-config prediction_length (official_pred_len), but
NOT actually true of the labels gift_eval.data.Dataset generates: for a
"long"-term config on a dataset whose series are short relative to the
nominal length, the split()/generate_instances() window near the tail of a
series is genuinely shorter than the nominal prediction_length (verified
directly: gift_eval's own TestData.label lengths for kdd_cup_2018_with_
missing/D/long are a MIX of 54 and 450 across its 270 items, not uniform).
A fixed-length forecast fails a numpy broadcast against a shorter label.

So instead: group items by their *actual* label length, batch-forecast
each group at that exact length (still batched per length group for GPU
efficiency, via eval_gift.forecast_rollout_batched, reused unmodified), and
hand the assembled per-item forecasts to gluonts.model.evaluate_forecasts
directly rather than evaluate_model. Every metric that only needs the 0.5
quantile (MASE, MAE, sMAPE, MSE, RMSE, NRMSE, ND) is computed by gluonts's
own real aggregation regardless of which of these two paths produced the
forecasts.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.split import TestData
from gluonts.itertools import batcher
from gluonts.model.evaluation import _get_data_batch
from gluonts.model.forecast import Forecast, QuantileForecast

_ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
for _sub in ("config", "lora", "clients", "hypernet", "data_generation"):
    _p = os.path.join(_ROOT_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_EVAL_DIR = os.path.dirname(__file__)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

import eval_gift as _eg  # noqa: E402  (reused for forecast_rollout_batched, model-agnostic)


def _label_length(label_entry: dict) -> int:
    t = np.asarray(label_entry["target"])
    return int(t.shape[-1]) if t.ndim > 0 else int(t.size)


def build_forecasts(
    model,
    test_data: TestData,
    context_length: int,
    step_horizon: int,
    device: torch.device,
    chunk: int = 128,
) -> Tuple[List[Forecast], TestData]:
    """Returns (forecasts, test_data) ready for gluonts.model.evaluate_forecasts.

    forecasts[i] corresponds to the i-th item of test_data.input / .label,
    each forecast exactly as long as that item's real label -- not
    necessarily the config's nominal prediction_length.
    """
    inputs = list(test_data.input)
    labels = list(test_data.label)
    assert len(inputs) == len(labels), (len(inputs), len(labels))

    groups: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[_label_length(lbl)].append(i)

    forecasts: List[Forecast] = [None] * len(inputs)  # type: ignore[list-item]
    for length, idxs in groups.items():
        if length <= 0:
            for i in idxs:
                inp = inputs[i]
                forecasts[i] = QuantileForecast(
                    forecast_arrays=np.zeros((1, 0), dtype=np.float32),
                    forecast_keys=["0.5"],
                    start_date=inp["start"] + len(inp["target"]),
                    item_id=inp.get("item_id"),
                )
            continue

        for s0 in range(0, len(idxs), chunk):
            batch_idxs = idxs[s0 : s0 + chunk]
            ctxs = np.zeros((len(batch_idxs), context_length), dtype=np.float32)
            for j, i in enumerate(batch_idxs):
                target = np.asarray(inputs[i]["target"], dtype=np.float32).reshape(-1)
                target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
                tail = target[-context_length:]
                ctxs[j, context_length - len(tail) :] = tail

            preds = _eg.forecast_rollout_batched(model, ctxs, length, device, step_horizon)

            for j, i in enumerate(batch_idxs):
                inp = inputs[i]
                forecasts[i] = QuantileForecast(
                    forecast_arrays=preds[j][None, :].astype(np.float32),
                    forecast_keys=["0.5"],
                    start_date=inp["start"] + len(inp["target"]),
                    item_id=inp.get("item_id"),
                )

    return forecasts, test_data


def evaluate_dataset_variable_length(
    model,
    test_data: TestData,
    metrics,
    context_length: int,
    step_horizon: int,
    device: torch.device,
    seasonality: Optional[int] = None,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    rollout_chunk: int = 128,
    gluonts_batch_size: int = 100,
) -> pd.DataFrame:
    """gluonts.model.evaluate_forecasts, but tolerant of per-item label
    lengths that vary *within one config* (verified real: kdd_cup_2018_with
    _missing/D/long's 270 test items split 54/450 between two lengths).

    gluonts's own evaluate_forecasts_raw batches input/label/forecast
    together and np.stack()s each batch's labels -- which crashes on a
    mixed-length batch, and does not reorder items to avoid one. Each
    DirectMetric's Mean(axis=...) aggregator is a streaming accumulator
    or... its .update(data_batch) is called once per (input, label,
    forecast) batch and .get() only combines those partial sums at the very
    end, addition being order-independent -- so replicating that loop but
    feeding it length-homogeneous batches (regrouped by actual label
    length rather than natural dataset order) produces the exact same
    axis=None global aggregate gluonts would produce if _get_data_batch
    could handle mixed lengths directly, just without ever handing it one.
    """
    inputs = list(test_data.input)
    labels = list(test_data.label)
    assert len(inputs) == len(labels), (len(inputs), len(labels))

    label_ndim = np.asarray(labels[0]["target"]).ndim
    axis = tuple(range(label_ndim + 1))  # matches evaluate_forecasts_raw's axis=None resolution
    evaluators = {}
    for metric in metrics:
        evaluator = metric(axis=axis)
        evaluators[evaluator.name] = evaluator

    groups: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[_label_length(lbl)].append(i)

    for length, idxs in groups.items():
        if length <= 0:
            continue
        for s0 in range(0, len(idxs), rollout_chunk):
            batch_idxs = idxs[s0 : s0 + rollout_chunk]
            ctxs = np.zeros((len(batch_idxs), context_length), dtype=np.float32)
            for j, i in enumerate(batch_idxs):
                target = np.asarray(inputs[i]["target"], dtype=np.float32).reshape(-1)
                target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
                tail = target[-context_length:]
                ctxs[j, context_length - len(tail) :] = tail
            preds = _eg.forecast_rollout_batched(model, ctxs, length, device, step_horizon)
            forecast_batch = [
                QuantileForecast(
                    forecast_arrays=preds[j][None, :].astype(np.float32),
                    forecast_keys=["0.5"],
                    start_date=inputs[i]["start"] + len(inputs[i]["target"]),
                    item_id=inputs[i].get("item_id"),
                )
                for j, i in enumerate(batch_idxs)
            ]

            # gluonts's own sub-batching within this (already length-uniform)
            # group -- irrelevant to correctness, only to _get_data_batch's
            # internal memory footprint.
            for gs0 in range(0, len(batch_idxs), gluonts_batch_size):
                sl = slice(gs0, gs0 + gluonts_batch_size)
                data_batch = _get_data_batch(
                    [inputs[i] for i in batch_idxs[sl]],
                    [labels[i] for i in batch_idxs[sl]],
                    forecast_batch[sl],
                    seasonality=seasonality,
                    mask_invalid_label=mask_invalid_label,
                    allow_nan_forecast=allow_nan_forecast,
                )
                for evaluator in evaluators.values():
                    evaluator.update(data_batch)

    metrics_values = {name: evaluator.get() for name, evaluator in evaluators.items()}
    flattened = {name: np.ravel(val) for name, val in metrics_values.items()}
    return pd.DataFrame(flattened, index=[None])
