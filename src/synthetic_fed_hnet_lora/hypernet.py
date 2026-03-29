import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNet(nn.Module):
    """Slightly stronger MLP 
    adds LayerNormand dropout because synthetic-regime experiments tend to overfit the
    client embedding table quickly.
    """

    def __init__(self, emb_dim: int, hidden: int, flat_dim: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, flat_dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v)


def mse_match_loss(pred_flat: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_flat, target_flat)
