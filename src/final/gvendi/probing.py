"""Adapter-gradient probing against a frozen restored checkpoint.

A probe is a forward/backward gradient *measurement* on a fixed model state:
nothing here updates the adapter, the forecast head, the frozen backbone, the
hypernetwork, or any optimizer. The one exception is the learnability test,
which runs a short throwaway local adaptation via the ordinary client-training
path and discards the adapted state.
"""

from __future__ import annotations

import os
import sys
import zlib

import torch
from torch.utils.data import Dataset, default_collate

_ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
for _sub in ("clients", "lora"):
    _p = os.path.join(_ROOT_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from client import (  # noqa: E402
    _extract_xy,
    _expand_stats_like,
    _maybe_match_target_shape,
    _move_batch_to_device,
    _run_model_forecast,
    evaluate_forecast,
    forecast_loss,
    load_forecast_head_state_dict,
    local_train_lora_and_head_steps,
)
from lora_utils import (  # noqa: E402
    inject_lora,
    load_flat_lora_into_model,
    mark_only_lora_trainable,
)

from .grad_vendi import RademacherProjector  # noqa: E402


def stable_probe_seed(probe_seed: int, client_key: str) -> int:
    """Deterministic per-client probe seed, independent of iteration order."""
    return (int(probe_seed) * 1_000_003 + zlib.crc32(client_key.encode())) % (2**31)


def make_probe_batches(
    dataset: Dataset,
    n_batches: int,
    batch_size: int,
    seed: int,
) -> list:
    """Draw deterministic probe batches from a dataset.

    Indices are sampled without replacement when the dataset is large enough
    (falling back to with-replacement for tiny probe datasets), then grouped
    into ``n_batches`` collated batches. The same (dataset, seed) pair always
    yields identical batches, so Random / Hard / H-Vendi runs share probes.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("cannot draw probe batches from an empty dataset")
    rng = np.random.default_rng(seed)
    needed = n_batches * batch_size
    if n >= needed:
        idx = rng.choice(n, size=needed, replace=False)
    else:
        idx = rng.choice(n, size=needed, replace=True)
    batches = []
    for b in range(n_batches):
        rows = [dataset[int(i)] for i in idx[b * batch_size:(b + 1) * batch_size]]
        batches.append(default_collate(rows))
    return batches


class ProbeEngine:
    """One reusable frozen model for gradient probes and proxy evaluation.

    The base model is built once; per-client work only swaps the flat LoRA
    adapter (and, temporarily, head state for the learnability test), which
    avoids reconstructing the backbone hundreds of times during a sweep.
    """

    def __init__(
        self,
        base_ctor,
        base_state: dict,
        spec,
        flatdim: int,
        global_head_state: dict,
        lora_cfg: dict,
        device,
        loss_type: str = "mae",
    ) -> None:
        self.spec = spec
        self.flatdim = int(flatdim)
        self.device = device
        self.loss_type = loss_type
        self.global_head_state = global_head_state

        model = base_ctor().to(device)
        model.load_state_dict(base_state, strict=True)
        load_forecast_head_state_dict(model, global_head_state)
        inject_lora(
            model,
            r=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            exclude_keywords=lora_cfg["exclude_keywords"],
        )
        mark_only_lora_trainable(model)
        model.eval()  # deterministic probes: no backbone/LoRA dropout
        self.model = model
        self._params = dict(model.named_parameters())

    # ------------------------------------------------------------------
    # State swapping
    # ------------------------------------------------------------------

    def set_adapter(self, flat: torch.Tensor) -> None:
        load_flat_lora_into_model(self.model, self.spec, flat)

    def load_head(self, head_state: dict) -> None:
        load_forecast_head_state_dict(self.model, head_state)

    def restore_global_head(self) -> None:
        load_forecast_head_state_dict(self.model, self.global_head_state)

    # ------------------------------------------------------------------
    # Loss / gradient measurement
    # ------------------------------------------------------------------

    def _batch_loss(self, batch) -> torch.Tensor:
        batch = _move_batch_to_device(batch, self.device)
        x, y, extras = _extract_xy(batch)
        yhat = _run_model_forecast(self.model, x, mask=extras.get("mask"))
        yhat, y = _maybe_match_target_shape(yhat, y)

        y_orig, mu, sd = extras.get("y_orig"), extras.get("mu"), extras.get("sd")
        if y_orig is not None and mu is not None and sd is not None:
            mu_exp = _expand_stats_like(mu, yhat)
            sd_exp = _expand_stats_like(sd, yhat)
            yhat_loss = yhat * sd_exp + mu_exp
            y_loss = y_orig
            yhat_loss, y_loss = _maybe_match_target_shape(yhat_loss, y_loss)
        else:
            yhat_loss, y_loss = yhat, y
        return forecast_loss(yhat_loss, y_loss, loss_type=self.loss_type)

    def _flatten_lora_grads(self) -> torch.Tensor:
        chunks = []
        for name, _shape, numel in self.spec:
            g = self._params[name].grad
            if g is None:
                chunks.append(torch.zeros(numel, dtype=torch.float32))
            else:
                chunks.append(g.detach().reshape(-1).to(torch.float32).cpu())
        return torch.cat(chunks, dim=0)

    def _zero_grads(self) -> None:
        for p in self.model.parameters():
            p.grad = None

    def probe_gradients(self, flat: torch.Tensor, batches: list) -> torch.Tensor:
        """Adapter gradients for each probe batch: [B, flatdim], detached.

        Pure measurement — no parameter is updated and no autograd graph is
        retained in the returned tensor.
        """
        self.set_adapter(flat)
        grads = []
        for batch in batches:
            self._zero_grads()
            loss = self._batch_loss(batch)
            loss.backward()
            grads.append(self._flatten_lora_grads())
        self._zero_grads()
        return torch.stack(grads, dim=0)

    @torch.no_grad()
    def loss_on_batches(self, flat: torch.Tensor, batches: list) -> float:
        self.set_adapter(flat)
        losses = [float(self._batch_loss(batch).item()) for batch in batches]
        return float(np.mean(losses))

    @torch.no_grad()
    def evaluate(
        self,
        flat: torch.Tensor,
        loader,
        max_batches: int | None = None,
        mase_denom: float | None = None,
        seasonal_naive_mape: float | None = None,
    ) -> dict:
        self.set_adapter(flat)
        return evaluate_forecast(
            self.model,
            loader,
            self.device,
            max_batches=max_batches,
            mase_denom=mase_denom,
            seasonal_naive_mape=seasonal_naive_mape,
        )

    # ------------------------------------------------------------------
    # Learnability (temporary local adaptation, state discarded)
    # ------------------------------------------------------------------

    def learnability_improvement(
        self,
        flat: torch.Tensor,
        train_loader,
        probe_batches: list,
        base_ctor,
        base_state: dict,
        local_steps: int,
        local_lr: float,
        lora_cfg: dict,
        eps: float = 1e-8,
    ) -> float:
        """I_c = (L_before - L_after) / (L_before + eps) after a short local fit."""
        loss_before = self.loss_on_batches(flat, probe_batches)
        updated_flat, updated_head = local_train_lora_and_head_steps(
            base_model_ctor=base_ctor,
            base_state_dict=base_state,
            spec=self.spec,
            init_lora_flat=flat,
            init_head_state=self.global_head_state,
            train_loader=train_loader,
            device=self.device,
            local_steps=local_steps,
            lr=local_lr,
            lora_cfg=lora_cfg,
            loss_type=self.loss_type,
        )
        try:
            self.load_head(updated_head)
            loss_after = self.loss_on_batches(updated_flat, probe_batches)
        finally:
            # Discard the adapted state entirely.
            self.restore_global_head()
        return float((loss_before - loss_after) / (loss_before + eps))


def collect_adapter_probe_gradients(
    engine: ProbeEngine,
    projector: RademacherProjector,
    adapters_and_datasets: dict[str, tuple[torch.Tensor, Dataset]],
    probe_batches_count: int,
    probe_batch_size: int,
    probe_seed: int,
) -> dict[str, torch.Tensor]:
    """Deterministic probe-gradient sweep.

    Args:
        adapters_and_datasets: client_key -> (flat adapter, probe dataset).

    Returns:
        client_key -> [B, d] projected, unit-norm probe gradients (CPU).
    """
    out: dict[str, torch.Tensor] = {}
    for key in sorted(adapters_and_datasets.keys()):
        flat, dataset = adapters_and_datasets[key]
        batches = make_probe_batches(
            dataset,
            n_batches=probe_batches_count,
            batch_size=probe_batch_size,
            seed=stable_probe_seed(probe_seed, key),
        )
        raw = engine.probe_gradients(flat, batches)
        if bool((raw.norm(dim=1) < 1e-12).any()):
            print(f"[gvendi][WARNING] zero probe gradient encountered for {key}")
        out[key] = projector.project(raw)
    return out
