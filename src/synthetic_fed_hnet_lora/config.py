from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model_id: str = "AutonLab/MOMENT-1-small"
    horizon: int = 32
    n_channels: int = 1

    # Overwritten
    n_clients: int = 54
    total_steps: int = 8000

    # ------------------------------------------------------------
    # Data source
    # ------------------------------------------------------------
    data_source: str = "synthetic"   # {"synthetic", "gift"}

    # ------------------------------------------------------------
    # Split / evaluation protocol
    # ------------------------------------------------------------
    test_frac: float = 0.10
    train_frac: float = 0.90
    val_frac: float = 0.0

    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------
    batch_size: int = 32
    num_workers: int = 0

    # ------------------------------------------------------------
    # Per-series normalization (train-only stats)
    # ------------------------------------------------------------
    normalize_per_series: bool = True
    normalization_eps: float = 1e-6
    clip_scale_min: float = 1e-4

    # ------------------------------------------------------------
    # Client local training
    # ------------------------------------------------------------
    local_steps: int = 70
    local_lr: float = 2e-3
    train_loss: str = "mae"   # {"mse", "mae", "smape"}

    # ------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    exclude_keywords: tuple = ("classifier",)

    # ------------------------------------------------------------
    # Federated
    # ------------------------------------------------------------
    rounds: int = 70
    clients_per_round: int = 8

    # ------------------------------------------------------------
    # Hypernetwork
    # ------------------------------------------------------------
    emb_dim: int = 17
    hnet_hidden: int = 256
    server_lr: float = 2e-3
    emb_batches: int = 32
    hnet_dropout: float = 0.10
    use_fixed_client_features: bool = True
    normalize_client_features: bool = True

    # ------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------
    synthetic_num_clients: int = 24
    synthetic_num_regimes: int = 6
    synthetic_variants_per_regime: int = 4
    synthetic_length: int = 5000
    synthetic_series_per_client: int = 16
    synthetic_series_length: int = 768
    synthetic_context_margin: int = 64
    synthetic_mixup_per_client: int = 2
    synthetic_max_mix_components: int = 2
    synthetic_mixup_alpha: float = 0.4
    synthetic_gp_samples_per_client: int = 4
    synthetic_kernel_terms_min: int = 1
    synthetic_kernel_terms_max: int = 2
    synthetic_kernel_noise: float = 0.02
    synthetic_use_mixup: bool = True
    synthetic_use_kernel_synth: bool = True

    # ------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------
    eval_every: int = 1
    eval_batches: int = 10
    mase_seasonality: int = 1

    # ------------------------------------------------------------
    # Repro / output
    # ------------------------------------------------------------
    seed: int = 0
    out_dir: str = "fed_hnet_lora_synth"

    # ------------------------------------------------------------
    # Synthetic regime feature names (metadata only)
    # ------------------------------------------------------------
    synthetic_feature_names: tuple = field(
        default_factory=lambda: (
            "trend_slope",
            "exp_trend_strength",
            "season_period_1",
            "season_amp_1",
            "season_period_2",
            "season_amp_2",
            "ar_coef_1",
            "ar_coef_2",
            "ar_coef_3",
            "noise_scale",
            "heteroskedasticity",
            "level_shift",
            "piecewise_trend",
            "kernel_linear",
            "kernel_rbf",
            "kernel_periodic",
            "kernel_noise",
        )
    )