import os
import random
from typing import Optional

import torch

from contrastive_utils import supervised_contrastive_loss
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
        self.grad_tracker = None  # optional GradVarianceTracker, set externally

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        learned = self.emb(ids)
        if self.client_features is None:
            return learned
        if self.learnable_embeddings:
            return learned + self.client_features[ids]
        return self.client_features[ids]

    def update_from_adapters(
        self,
        client_ids,
        adapter_targets_cpu,
        contrastive_labels=None,
        use_contrastive_loss: bool = False,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.2,
        contrastive_mode: str = "supervised_regime",
        return_details: bool = False,
    ):
        ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)
        embeddings = self._embed(ids)
        pred = self.hnet(embeddings)
        target = torch.stack([adapter_targets_cpu[i].to(self.device) for i in client_ids], dim=0)

        base_loss = mse_match_loss(pred, target)
        contrastive_loss = pred.new_zeros(())
        contrastive_info = {
            "contrastive_batch_size": int(len(client_ids)),
            "contrastive_positive_anchors": 0,
            "contrastive_positive_pairs": 0,
            "contrastive_temperature": float(contrastive_temperature),
            "contrastive_mode": contrastive_mode,
        }

        if use_contrastive_loss:
            if contrastive_labels is None:
                raise ValueError("contrastive_labels are required when use_contrastive_loss=True")
            contrastive_loss, contrastive_info = supervised_contrastive_loss(
                embeddings=embeddings,
                labels=torch.as_tensor(contrastive_labels, device=self.device),
                temperature=contrastive_temperature,
                mode=contrastive_mode,
            )

        total_loss = base_loss + float(contrastive_weight) * contrastive_loss

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.grad_tracker is not None:
            self.grad_tracker.update(self.hnet)
        self.opt.step()

        out = {
            "server_target_loss": float(base_loss.item()),
            "server_contrastive_loss": float(contrastive_loss.item()),
            "server_contrastive_weighted_loss": float(
                float(contrastive_weight) * contrastive_loss.item()
            ),
            "server_total_loss": float(total_loss.item()),
        }
        out.update(contrastive_info)
        if return_details:
            return out
        return out["server_target_loss"]

    def update_from_targets(self, client_ids, target_flats_cpu):
        ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)
        pred = self.hnet(self._embed(ids))
        target = torch.stack([target_flats_cpu[i].to(self.device) for i in client_ids], dim=0)
        loss = mse_match_loss(pred, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_tracker is not None:
            self.grad_tracker.update(self.hnet)
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
        was_training = self.hnet.training
        self.hnet.eval()
        try:
            return self.hnet(feats)
        finally:
            if was_training:
                self.hnet.train()

    @torch.no_grad()
    def generate_lora_flat(self, client_ids):
        ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)
        was_training = self.hnet.training
        self.hnet.eval()
        try:
            return self.hnet(self._embed(ids))
        finally:
            if was_training:
                self.hnet.train()

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
