import torch
import torch.nn as nn
import torch.nn.functional as F

#just a mlp. 
class HyperNet(nn.Module):
    def __init__(self, emb_dim: int, hidden: int, flat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, flat_dim),
        )

    def forward(self, v):
        return self.net(v)


#TODO: experiment with this stuff. 
def mse_match_loss(pred_flat, target_flat):
    return F.mse_loss(pred_flat, target_flat)
