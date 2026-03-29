import copy

import torch
import torch.nn.functional as F

from lora_utils import (
    flatten_lora,
    inject_lora,
    mark_only_lora_trainable,
    load_flat_lora_into_model,
)


def _move_obj_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_move_obj_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _move_obj_to_device(v, device) for k, v in x.items()}
    return x


def _move_batch_to_device(batch, device):
    return _move_obj_to_device(batch, device)


def _extract_from_dict_batch(batch):
    x = None
    y = None

    for k in ["x_enc", "history", "past_values", "x", "encoder_input"]:
        if k in batch and torch.is_tensor(batch[k]):
            x = batch[k]
            break

    for k in ["y", "labels", "target", "future_values", "y_true"]:
        if k in batch and torch.is_tensor(batch[k]):
            y = batch[k]
            break

    extras = {
        "mask": batch.get("mask", None),
        "y_orig": batch.get("y_orig", None),
        "mu": batch.get("mu", None),
        "sd": batch.get("sd", None),
    }
    return x, y, extras


def _extract_from_sequence_batch(batch):
    if len(batch) < 2:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    x = batch[0]
    y = batch[1]
    extras = {
        "mask": batch[2] if len(batch) >= 3 and torch.is_tensor(batch[2]) else None,
        "y_orig": batch[3] if len(batch) >= 4 and torch.is_tensor(batch[3]) else None,
        "mu": batch[4] if len(batch) >= 5 and torch.is_tensor(batch[4]) else None,
        "sd": batch[5] if len(batch) >= 6 and torch.is_tensor(batch[5]) else None,
    }
    return x, y, extras


def _extract_xy(batch):
    if isinstance(batch, dict):
        x, y, extras = _extract_from_dict_batch(batch)
    elif isinstance(batch, (list, tuple)):
        x, y, extras = _extract_from_sequence_batch(batch)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if x is None or y is None:
        raise ValueError(f"Could not extract x/y from batch of type {type(batch)}")
    return x, y, extras


def _run_model_forecast(model, x, mask=None):
    out = model(x_enc=x)
    if hasattr(out, "forecast"):
        return out.forecast
    if torch.is_tensor(out):
        return out
    raise ValueError("Model output does not contain `.forecast` and is not a tensor.")


def _maybe_match_target_shape(yhat, y):
    if yhat.shape == y.shape:
        return yhat, y
    if yhat.ndim == y.ndim + 1 and yhat.shape[-1] == 1:
        yhat2 = yhat.squeeze(-1)
        if yhat2.shape == y.shape:
            return yhat2, y
    if y.ndim == yhat.ndim + 1 and y.shape[-1] == 1:
        y2 = y.squeeze(-1)
        if y2.shape == yhat.shape:
            return yhat, y2
    return yhat, y


def _expand_stats_like(stat, target):
    if stat is None:
        return None
    while stat.ndim < target.ndim:
        stat = stat.unsqueeze(-1)
    return stat


def smape_loss(yhat, y, eps=1e-8):
    return (2.0 * (yhat - y).abs() / (yhat.abs() + y.abs() + eps)).mean()


def forecast_loss(yhat, y, loss_type="smape"):
    if loss_type == "mse":
        return F.mse_loss(yhat, y)
    if loss_type == "mae":
        return F.l1_loss(yhat, y)
    if loss_type == "smape":
        return smape_loss(yhat, y)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def is_forecast_head_param(name: str) -> bool:
    n = name.lower()
    if "classifier" in n:
        return False
    return ("forecast" in n) or ("head" in n)


def extract_forecast_head_state_dict(model):
    out = {}
    for name, tensor in model.state_dict().items():
        if is_forecast_head_param(name):
            out[name] = tensor.detach().cpu().clone()
    return out


def load_forecast_head_state_dict(model, head_state_dict):
    if head_state_dict is None:
        return
    model_sd = model.state_dict()
    for name, tensor in head_state_dict.items():
        if name in model_sd:
            model_sd[name].copy_(tensor.to(model_sd[name].device, dtype=model_sd[name].dtype))


def enable_forecast_head_training(model):
    found = []
    for name, param in model.named_parameters():
        if is_forecast_head_param(name):
            param.requires_grad = True
            found.append(name)
    if not found:
        raise RuntimeError(
            "No forecast-head parameters found. Print model.named_parameters() and adjust is_forecast_head_param()."
        )
    return found


def local_train_lora_and_head_steps(
    base_model_ctor,
    base_state_dict,
    spec,
    init_lora_flat,
    init_head_state,
    train_loader,
    device,
    local_steps,
    lr,
    lora_cfg,
    loss_type="smape",
):
    model = base_model_ctor().to(device)
    model.load_state_dict(base_state_dict, strict=True)

    # Load shared forecast-head weights before local training
    load_forecast_head_state_dict(model, init_head_state)

    inject_lora(
        model,
        r=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        exclude_keywords=lora_cfg["exclude_keywords"],
    )
    mark_only_lora_trainable(model)
    enable_forecast_head_training(model)

    load_flat_lora_into_model(model, spec, init_lora_flat)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")

    opt = torch.optim.AdamW(params, lr=lr)
    model.train()

    steps_done = 0
    train_iter = iter(train_loader)

    while steps_done < local_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = _move_batch_to_device(batch, device)
        x, y, extras = _extract_xy(batch)
        mask = extras.get("mask", None)

        opt.zero_grad(set_to_none=True)
        yhat = _run_model_forecast(model, x, mask=mask)
        yhat, y = _maybe_match_target_shape(yhat, y)

        y_orig = extras.get("y_orig", None)
        mu = extras.get("mu", None)
        sd = extras.get("sd", None)

        if y_orig is not None and mu is not None and sd is not None:
            mu_exp = _expand_stats_like(mu, yhat)
            sd_exp = _expand_stats_like(sd, yhat)
            yhat_loss = yhat * sd_exp + mu_exp
            y_loss = y_orig
            yhat_loss, y_loss = _maybe_match_target_shape(yhat_loss, y_loss)
        else:
            yhat_loss = yhat
            y_loss = y

        loss = forecast_loss(yhat_loss, y_loss, loss_type=loss_type)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss in local_train_lora_and_head_steps: loss_type={loss_type}, step={steps_done}"
            )
        loss.backward()
        opt.step()
        steps_done += 1

    updated_lora_flat = flatten_lora(model, spec).detach().cpu()
    updated_head_state = extract_forecast_head_state_dict(model)
    return updated_lora_flat, updated_head_state


@torch.no_grad()
def evaluate_forecast(
    model,
    loader,
    device,
    max_batches=None,
    mase_denom=None,
    seasonal_naive_mape=None,
    eps=1e-8,
):
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    smape_sum = 0.0
    n = 0
    ape_per_series = []

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        batch = _move_batch_to_device(batch, device)
        x, y, extras = _extract_xy(batch)
        mask = extras.get("mask", None)

        yhat = _run_model_forecast(model, x, mask=mask)
        yhat, y = _maybe_match_target_shape(yhat, y)

        y_orig = extras.get("y_orig", None)
        mu = extras.get("mu", None)
        sd = extras.get("sd", None)

        if y_orig is not None and mu is not None and sd is not None:
            mu = _expand_stats_like(mu, yhat)
            sd = _expand_stats_like(sd, yhat)
            yhat_orig = yhat * sd + mu
            y_true = y_orig
        else:
            yhat_orig = yhat
            y_true = y

        yhat_orig, y_true = _maybe_match_target_shape(yhat_orig, y_true)
        err = yhat_orig - y_true

        se_sum += (err ** 2).sum().item()
        ae_sum += err.abs().sum().item()
        smape_sum += (2.0 * err.abs() / (yhat_orig.abs() + y_true.abs() + eps)).sum().item()
        n += y_true.numel()

        if y_true.ndim == 3 and y_true.shape[1] == 1:
            yt = y_true.squeeze(1)
            yp = yhat_orig.squeeze(1)
        else:
            yt = y_true.reshape(y_true.shape[0], -1)
            yp = yhat_orig.reshape(yhat_orig.shape[0], -1)

        ape = ((yp - yt).abs() / (yt.abs() + eps)).mean(dim=1)
        ape_per_series.extend(ape.detach().cpu().tolist())

    mse = se_sum / max(n, 1)
    mae = ae_sum / max(n, 1)
    smape = smape_sum / max(n, 1)
    median_mape = float(torch.tensor(ape_per_series).median().item()) if ape_per_series else None

    out = {
        "mse": mse,
        "mae": mae,
        "smape": smape,
        "median_mape": median_mape,
        "se_sum": se_sum,
        "ae_sum": ae_sum,
        "smape_sum": smape_sum,
        "n": n,
    }
    if mase_denom is not None:
        out["mase"] = mae / max(mase_denom, eps)
    if seasonal_naive_mape is not None and median_mape is not None:
        out["norm_mape"] = median_mape / max(seasonal_naive_mape, eps)
    return out