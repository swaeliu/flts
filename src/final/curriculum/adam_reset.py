"""ReLoRA-style magnitude-based partial reset of Adam optimizer moments.

At each curriculum round boundary the accumulated first (m) and second (v)
moment estimates carry directional memory biased toward the round-k loss
landscape.  Zeroing the lowest-magnitude entries clears stale momentum
without the instability of a full wipe.

References
----------
Lialin et al., "ReLoRA: High-Rank Training Through Low-Rank Updates" (2023).
"""

from __future__ import annotations

import torch


def prune_adam_moments(
    optimizer: torch.optim.Optimizer,
    pruning_pct: float,
) -> torch.optim.Optimizer:
    """Zero out the smallest-magnitude Adam moments and set all LRs to zero.

    For every parameter that has accumulated state, the absolute value of each
    entry in the first-moment (exp_avg / m) buffer is used to determine a
    magnitude threshold at the *pruning_pct* percentile.  Entries at or below
    that threshold are zeroed in both the m and v buffers.  Parameters with no
    accumulated state (e.g. freshly added embedding rows) are left untouched.

    After pruning, the learning rate for every param group is set to 0.0 so
    that training can be resumed with a linear warmup.

    Args:
        optimizer: An AdamW / Adam optimizer whose state may contain
            ``exp_avg`` and ``exp_avg_sq`` buffers.
        pruning_pct: Fraction of entries to zero, in [0.0, 1.0).
            0.9 zeros the bottom 90 % by |m|; 0.0 is a no-op.

    Returns:
        The same optimizer object (modified in-place).
    """
    if not (0.0 <= pruning_pct < 1.0):
        raise ValueError(f"pruning_pct must be in [0, 1), got {pruning_pct}")

    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is None or "exp_avg" not in state:
                continue

            m: torch.Tensor = state["exp_avg"]
            v: torch.Tensor = state["exp_avg_sq"]

            # Threshold from absolute values of first moments.
            # torch.quantile is limited to 2^24 elements; use kthvalue instead.
            m_flat = m.abs().float().flatten()
            k = max(1, int(pruning_pct * m_flat.numel()))
            threshold = torch.kthvalue(m_flat, k).values

            # Retain entries strictly above the threshold.
            keep_mask = (m.abs() > threshold).to(m.dtype)
            m.mul_(keep_mask)
            v.mul_(keep_mask)

        # LR to zero; warmup will ramp it back up.
        group["lr"] = 0.0

    return optimizer


def full_adam_reset(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Wipe all Adam optimizer state (exp_avg, exp_avg_sq, step) and zero LRs.

    Unlike ``prune_adam_moments``, which cannot reach 100% (it requires
    ``pruning_pct < 1.0``) and only zeros the moment tensors in place, this
    deletes each parameter's state entry outright. That also resets the
    internal step counter, so Adam's bias-correction restarts from scratch
    rather than continuing under an elevated step count with zeroed moments.

    Args:
        optimizer: An AdamW / Adam optimizer whose state may contain
            ``exp_avg`` / ``exp_avg_sq`` buffers.

    Returns:
        The same optimizer object (modified in-place).
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                del optimizer.state[p]
        group["lr"] = 0.0

    return optimizer


def partial_adam_reset(
    optimizer: torch.optim.Optimizer,
    importance_scores: dict[torch.nn.Parameter, torch.Tensor],
    retain_fraction: float = 0.10,
) -> torch.optim.Optimizer:
    """Zero Adam moments for parameters below a global importance threshold.

    All tensors in *importance_scores* are flattened into a single vector to
    find the global ``(1 - retain_fraction)`` quantile, used as a scalar
    threshold. For every tracked parameter with accumulated Adam state, the
    ``exp_avg`` and ``exp_avg_sq`` buffers are multiplied elementwise by a
    binary mask that keeps only entries whose importance magnitude exceeds
    the threshold — zeroing the bottom ``1 - retain_fraction`` of moments.

    Args:
        optimizer: An AdamW / Adam optimizer whose state may contain
            ``exp_avg`` and ``exp_avg_sq`` buffers.
        importance_scores: Per-parameter importance tensors (same shape as
            the parameter), e.g. ``GradVarianceTracker.variance()`` or
            ``.snr()``.
        retain_fraction: Fraction of entries to keep, in (0.0, 1.0].
            0.10 keeps the top 10 % by importance magnitude.

    Returns:
        The same optimizer object (modified in-place).
    """
    if not (0.0 < retain_fraction <= 1.0):
        raise ValueError(f"retain_fraction must be in (0, 1], got {retain_fraction}")

    flat = torch.cat(
        [score.detach().abs().float().flatten() for score in importance_scores.values()]
    )
    k = max(1, int((1.0 - retain_fraction) * flat.numel()))
    threshold = torch.kthvalue(flat, k).values

    for p, score in importance_scores.items():
        state = optimizer.state.get(p)
        if state is None or "exp_avg" not in state:
            continue

        mask = (score.abs() > threshold).to(state["exp_avg"].dtype)
        state["exp_avg"].mul_(mask)
        state["exp_avg_sq"].mul_(mask)

    return optimizer


def count_nonzero_moments(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Diagnostic: count non-zero entries in m buffers per param group name."""
    counts: dict[str, int] = {}
    for group in optimizer.param_groups:
        name = group.get("name", "unnamed")
        total, nonzero = 0, 0
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state and "exp_avg" in state:
                m = state["exp_avg"]
                total += m.numel()
                nonzero += int((m != 0).sum().item())
        counts[name] = {"total": total, "nonzero": nonzero}
    return counts
