import os
import random
import torch
import torch.nn.functional as F


from hypernet import HyperNet, mse_match_loss

class Server:
    def __init__(self, n_clients, emb_dim, hidden, flat_dim, lr, device):
        self.device = device
        self.hnet = HyperNet(emb_dim=emb_dim, hidden=hidden, flat_dim=flat_dim).to(device)
        self.emb = torch.nn.Embedding(n_clients, emb_dim).to(device)
        torch.nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

        self.opt = torch.optim.AdamW(list(self.hnet.parameters()) + list(self.emb.parameters()), lr=lr)

    def update_from_deltas(self, client_ids, delta_targets_cpu):
        ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)
        v = self.emb(ids)          
        pred = self.hnet(v)         

        delta = torch.stack([delta_targets_cpu[i].to(self.device) for i in client_ids], dim=0)


        flat_dim = pred.shape[1]
        loss = - (pred * delta).sum(dim=1).mean() / float(flat_dim)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        alignment = float((pred.detach() * delta).sum(dim=1).mean().item() / float(flat_dim))
        return alignment


    def sample_clients(self, clients_per_round):
        n = self.emb.num_embeddings
        return random.sample(range(n), k=min(clients_per_round, n))

    def generate_lora_flat(self, client_ids):
        ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)
        v = self.emb(ids)
        pred = self.hnet(v)                  
        return pred

    def update_from_targets(self, client_ids, target_flats_cpu):
        ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)
        v = self.emb(ids)
        pred = self.hnet(v)  

        target = torch.stack([target_flats_cpu[i].to(self.device) for i in client_ids], dim=0)
        loss = mse_match_loss(pred, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "hnet": self.hnet.state_dict(),
                "emb": self.emb.state_dict(),
            },
            os.path.join(out_dir, "server.pt"),
        )
