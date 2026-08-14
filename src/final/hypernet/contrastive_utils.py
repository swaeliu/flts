from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
    mode: str = "supervised_regime",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute a supervised contrastive loss over a minibatch of client vectors."""
    if mode != "supervised_regime":
        raise ValueError(f"Unsupported contrastive_mode={mode!r}")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [batch, dim], got {tuple(embeddings.shape)}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    labels = torch.as_tensor(labels, device=embeddings.device)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if labels.numel() != embeddings.shape[0]:
        raise ValueError(
            f"labels must have one entry per embedding; got {labels.numel()} labels "
            f"for {embeddings.shape[0]} embeddings"
        )

    batch_size = embeddings.shape[0]
    zero = embeddings.sum() * 0.0
    if batch_size < 2:
        return zero, _diagnostics(batch_size, 0, 0, temperature, mode)

    z = F.normalize(embeddings.float(), p=2, dim=1, eps=eps)
    logits = torch.matmul(z, z.T) / float(temperature)

    eye = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    logits_mask = ~eye
    positives = labels[:, None].eq(labels[None, :]) & logits_mask

    positive_counts = positives.sum(dim=1)
    valid = positive_counts > 0
    positive_anchor_count = int(valid.sum().detach().cpu().item())
    positive_pair_count = int(positives.sum().detach().cpu().item())
    if positive_anchor_count == 0:
        return zero, _diagnostics(batch_size, 0, 0, temperature, mode)

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(eps))

    per_anchor_loss = -(log_prob * positives.float()).sum(dim=1) / positive_counts.clamp_min(1)
    loss = per_anchor_loss[valid].mean()
    return loss, _diagnostics(
        batch_size,
        positive_anchor_count,
        positive_pair_count,
        temperature,
        mode,
    )


def _diagnostics(
    batch_size: int,
    positive_anchor_count: int,
    positive_pair_count: int,
    temperature: float,
    mode: str,
) -> dict[str, Any]:
    return {
        "contrastive_batch_size": int(batch_size),
        "contrastive_positive_anchors": int(positive_anchor_count),
        "contrastive_positive_pairs": int(positive_pair_count),
        "contrastive_temperature": float(temperature),
        "contrastive_mode": mode,
    }
