"""Per-parameter running gradient statistics via Welford's online algorithm.

Tracks ``mean`` and ``M2`` (sum of squared deviations from the mean) for each
tracked parameter's gradient, without ever materialising a gradient history
buffer. Used to derive variance- and SNR-based importance scores for
:func:`relora.adam_reset.partial_adam_reset`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradVarianceTracker:
    """Welford-style running gradient mean/variance, tracked per parameter."""

    def __init__(self) -> None:
        self._mean: dict[nn.Parameter, torch.Tensor] = {}
        self._m2: dict[nn.Parameter, torch.Tensor] = {}
        self._n: int = 0

    def update(self, model: nn.Module) -> None:
        """Welford update from ``model``'s current ``.grad`` tensors.

        Call after ``backward()`` and before ``optimizer.step()``.
        """
        self._n += 1
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if p not in self._mean:
                self._mean[p] = torch.zeros_like(g)
                self._m2[p] = torch.zeros_like(g)
            mean = self._mean[p]
            delta = g - mean
            mean.add_(delta / self._n)
            self._m2[p].add_(delta * (g - mean))

    def variance(self) -> dict[nn.Parameter, torch.Tensor]:
        """Per-parameter unbiased variance, ``M2 / (n - 1)``."""
        if self._n < 2:
            return {p: torch.zeros_like(m2) for p, m2 in self._m2.items()}
        return {p: m2 / (self._n - 1) for p, m2 in self._m2.items()}

    def snr(self) -> dict[nn.Parameter, torch.Tensor]:
        """Per-parameter signal-to-noise ratio, ``mean^2 / (variance + 1e-8)``."""
        variance = self.variance()
        return {p: mean.pow(2) / (variance[p] + 1e-8) for p, mean in self._mean.items()}

    def reset(self) -> None:
        """Zero all accumulators (keeps tracked parameters, drops history)."""
        for p in self._mean:
            self._mean[p].zero_()
            self._m2[p].zero_()
        self._n = 0
