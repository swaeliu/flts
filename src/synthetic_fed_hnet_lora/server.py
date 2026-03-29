import os
import random
from typing import Optional

import torch

from hypernet import HyperNet, mse_match_loss



class Server:
    def __init__(
        self,
        n_clients: int,
        emb_dim: int,
        hidden: int,
        flat_dim: int,
        lr: float,
        device,
        client_features: Optional[torch.Tensor] = None,
        hnet_dropout: float = 0.10,
        learnable_embeddings: bool = True,
    ):
        self.device = device
        self.flat_dim = flat_dim
        self.hnet = HyperNet(
            emb_dim=emb_dim,
            hidden=hidden,
            flat_dim=flat_dim,
            dropout=hnet_dropout,
        ).to(device)

        self.learnable_embeddings = learnable_embeddings or client_features is None
        if client_features is not None:
            feats = client_features.detach().float().cpu()
            if feats.ndim != 2 or feats.shape[0] != n_clients:
                raise ValueError(
                    f"client_features must have shape [n_clients, emb_dim]; got {tuple(feats.shape)}"
                )
            if feats.shape[1] != emb_dim:
                raise ValueError(f"client_features dim {feats.shape[1]} != emb_dim {emb_dim}")
            self.client_features = feats.to(device)
        else:
            self.client_features = None

        self.emb = torch.nn.Embedding(n_clients, emb_dim).to(device)
        torch.nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

        opt_params = list(self.hnet.parameters())
        if self.learnable_embeddings:
            opt_params += list(self.emb.parameters())
        self.opt = torch.optim.AdamW(opt_params, lr=lr, weight_decay=1e-4)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        learned = self.emb(ids)
        if self.client_features is None:
            return learned
        if self.learnable_embeddings:
            return learned + self.client_features[ids]
        return self.client_features[ids]

    def update_from_deltas(self, client_ids, delta_targets_cpu):
        ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)
        pred = self.hnet(self._embed(ids))
        delta = torch.stack([delta_targets_cpu[i].to(self.device) for i in client_ids], dim=0)

        loss = mse_match_loss(pred, delta)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def update_from_targets(self, client_ids, target_flats_cpu):
        ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)
        pred = self.hnet(self._embed(ids))
        target = torch.stack([target_flats_cpu[i].to(self.device) for i in client_ids], dim=0)
        loss = mse_match_loss(pred, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def sample_clients(self, clients_per_round):
        n = self.emb.num_embeddings
        return random.sample(range(n), k=min(clients_per_round, n))
    
    @torch.no_grad()
    def generate_lora_flat_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B, emb_dim]
        Returns:  [B, flat_dim]
        """
        feats = features.to(self.device).float()
        return self.hnet(feats)

    @torch.no_grad()
    def generate_lora_flat(self, client_ids):
        ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)
        return self.hnet(self._embed(ids))

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "hnet": self.hnet.state_dict(),
            "emb": self.emb.state_dict(),
            "learnable_embeddings": self.learnable_embeddings,
        }
        if self.client_features is not None:
            payload["client_features"] = self.client_features.detach().cpu()
        torch.save(payload, os.path.join(out_dir, "server.pt"))