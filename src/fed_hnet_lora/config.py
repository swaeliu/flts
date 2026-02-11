from dataclasses import dataclass

@dataclass
class Config:
    #TODO make this bettter. 
    # Model
    model_id: str = "AutonLab/MOMENT-1-small"
    horizon: int = 32
    n_channels: int = 1

    # Synthetic data
    n_clients: int = 8
    total_steps: int = 6000
    train_frac: float = 0.70
    val_frac: float = 0.15

    # DataLoader
    batch_size: int = 32

    # Client local training
    local_steps: int = 20        
    local_lr: float = 2e-4

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    exclude_keywords: tuple = ("classifier",)

    # Federated
    rounds: int = 25
    clients_per_round: int = 4

    # Hypernetwork
    emb_dim: int = 64
    hnet_hidden: int = 256
    server_lr: float = 1e-3

    # Output
    out_dir: str = "fed_hnet_lora_out"
