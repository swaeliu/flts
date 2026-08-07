"""GVendi configuration, extending ReLoRAConfig with augmentation settings."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

_RELORA_DIR = os.path.join(os.path.dirname(__file__), "..")
if _RELORA_DIR not in sys.path:
    sys.path.insert(0, _RELORA_DIR)
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "synthetic_fed_hnet_lora")
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from relora.config import ReLoRAConfig  # noqa: E402

SELECTION_METHODS = ("random", "hard", "h_vendi")


@dataclass
class GVendiConfig(ReLoRAConfig):
    # ------------------------------------------------------------------
    # Backbone selection
    # ------------------------------------------------------------------
    # "moment" (default, unchanged behavior), "chronos" (see
    # gvendi_chronos/), or "patchtst" (see gvendi_patchtst/ -- a
    # from-scratch, randomly-initialized PatchTST, since no general-purpose
    # pretrained PatchTST checkpoint comparable to MOMENT/Chronos exists).
    # Read by run_gvendi_8x8._build_base_model. A stage loaded from a
    # previous stage's round_N_config.json inherits this automatically, so
    # it only needs to be set explicitly at the base stage that starts a
    # chain.
    backbone: str = "moment"

    # ------------------------------------------------------------------
    # H-Vendi augmentation
    # ------------------------------------------------------------------
    gvendi_enabled: bool = True
    gvendi_selection_method: str = "h_vendi"  # {"random", "hard", "h_vendi"}

    # ------------------------------------------------------------------
    # Gradient probes
    # ------------------------------------------------------------------
    gvendi_gradient_target: str = "adapter"  # only "adapter" in the first version
    gvendi_projection_dim: int = 256
    gvendi_projection_seed: int = 43
    gvendi_probe_seed: int = 44
    gvendi_probe_batches: int = 4
    gvendi_probe_batch_size: int = 32

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    gvendi_candidate_seed: int = 42
    gvendi_candidates_per_existing_regime: int = 16
    gvendi_candidates_new_regime: int = 128
    # Regime index of the fixed new regime; -1 means "prev_n_regimes"
    # (i.e. the next free index, 7 for the 7x7 -> 8x8 boundary).
    gvendi_new_regime_index: int = -1
    # Generator family for the new regime; "" keeps the legacy family cycle
    # (_regime_family(new_regime_index)). Must be fixed before generation.
    gvendi_new_regime_family: str = ""
    # Candidate-generation strategy mixture for existing regimes.
    gvendi_frac_perturb: float = 0.70
    gvendi_frac_interpolate: float = 0.20
    gvendi_frac_explore: float = 0.10
    gvendi_perturb_scale: float = 1.0

    # Probe-only data volume (full volume is generated only for selected clients).
    gvendi_probe_series_per_client: int = 4
    gvendi_probe_gp_samples: int = 2
    gvendi_probe_mixup_per_client: int = 1

    # ------------------------------------------------------------------
    # Quality gate
    # ------------------------------------------------------------------
    gvendi_mase_low_quantile: float = 0.50
    gvendi_mase_high_quantile: float = 0.95
    gvendi_learnability_steps: int = 5
    gvendi_min_learnability: float = 0.05
    gvendi_min_existing_feature_distance: float = 0.0
    gvendi_min_selected_feature_distance: float = 0.0

    # ------------------------------------------------------------------
    # Sparse-region shortlisting
    # ------------------------------------------------------------------
    gvendi_num_clusters: int = 16
    gvendi_sparse_cluster_fraction: float = 0.25
    gvendi_min_sparse_fraction: float = 0.25
    gvendi_centroid_distance_quantile: float = 0.75

    # ------------------------------------------------------------------
    # Greedy selection
    # ------------------------------------------------------------------
    gvendi_regime_order_trials: int = 16
    gvendi_selection_seed: int = 45  # drives Random baseline + regime orders

    # ------------------------------------------------------------------
    # Sampling after expansion (identical across selection methods)
    # ------------------------------------------------------------------
    gvendi_new_client_sampling_multiplier: float = 2.0
    gvendi_new_client_sampling_rounds: int = 20
    gvendi_new_client_batch_fraction: float = 0.50

    # ------------------------------------------------------------------
    # Cohort-aware checkpoint selection (diagnostic; only active when
    # run_gvendi_stage(..., track_checkpoint_variants=True))
    # ------------------------------------------------------------------
    # "constrained_best" tolerates at most this relative regression in
    # old-cohort mean val_mae (vs. the best old-cohort value seen so far in
    # the stage) while otherwise minimizing new-cohort mean val_mae.
    gvendi_forgetting_tolerance: float = 0.02

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    gvendi_cache_gradients: bool = True
    gvendi_resume_from_manifest: bool = True


def add_gvendi_cli_args(parser) -> None:
    """Register all gvendi_* overrides on an argparse parser (None = keep default)."""
    p = parser
    p.add_argument("--gvendi_selection_method", type=str, default=None,
                   choices=list(SELECTION_METHODS))
    p.add_argument("--gvendi_projection_dim", type=int, default=None)
    p.add_argument("--gvendi_projection_seed", type=int, default=None)
    p.add_argument("--gvendi_probe_seed", type=int, default=None)
    p.add_argument("--gvendi_probe_batches", type=int, default=None)
    p.add_argument("--gvendi_probe_batch_size", type=int, default=None)
    p.add_argument("--gvendi_candidate_seed", type=int, default=None)
    p.add_argument("--gvendi_candidates_per_existing_regime", type=int, default=None)
    p.add_argument("--gvendi_candidates_new_regime", type=int, default=None)
    p.add_argument("--gvendi_new_regime_index", type=int, default=None)
    p.add_argument("--gvendi_new_regime_family", type=str, default=None)
    p.add_argument("--gvendi_probe_series_per_client", type=int, default=None)
    p.add_argument("--gvendi_mase_low_quantile", type=float, default=None)
    p.add_argument("--gvendi_mase_high_quantile", type=float, default=None)
    p.add_argument("--gvendi_learnability_steps", type=int, default=None)
    p.add_argument("--gvendi_min_learnability", type=float, default=None)
    p.add_argument("--gvendi_min_existing_feature_distance", type=float, default=None)
    p.add_argument("--gvendi_min_selected_feature_distance", type=float, default=None)
    p.add_argument("--gvendi_num_clusters", type=int, default=None)
    p.add_argument("--gvendi_sparse_cluster_fraction", type=float, default=None)
    p.add_argument("--gvendi_min_sparse_fraction", type=float, default=None)
    p.add_argument("--gvendi_centroid_distance_quantile", type=float, default=None)
    p.add_argument("--gvendi_regime_order_trials", type=int, default=None)
    p.add_argument("--gvendi_selection_seed", type=int, default=None)
    p.add_argument("--gvendi_new_client_sampling_multiplier", type=float, default=None)
    p.add_argument("--gvendi_new_client_sampling_rounds", type=int, default=None)
    p.add_argument("--gvendi_new_client_batch_fraction", type=float, default=None)
    p.add_argument("--gvendi_forgetting_tolerance", type=float, default=None)


GVENDI_OVERRIDE_NAMES = [
    "gvendi_selection_method",
    "gvendi_projection_dim", "gvendi_projection_seed",
    "gvendi_probe_seed", "gvendi_probe_batches", "gvendi_probe_batch_size",
    "gvendi_candidate_seed", "gvendi_candidates_per_existing_regime",
    "gvendi_candidates_new_regime", "gvendi_new_regime_index",
    "gvendi_new_regime_family", "gvendi_probe_series_per_client",
    "gvendi_mase_low_quantile", "gvendi_mase_high_quantile",
    "gvendi_learnability_steps", "gvendi_min_learnability",
    "gvendi_min_existing_feature_distance", "gvendi_min_selected_feature_distance",
    "gvendi_num_clusters", "gvendi_sparse_cluster_fraction",
    "gvendi_min_sparse_fraction", "gvendi_centroid_distance_quantile",
    "gvendi_regime_order_trials", "gvendi_selection_seed",
    "gvendi_new_client_sampling_multiplier", "gvendi_new_client_sampling_rounds",
    "gvendi_new_client_batch_fraction", "gvendi_forgetting_tolerance",
]


def validate_gvendi_config(
    cfg: GVendiConfig,
    prev_n_regimes: int,
    prev_n_variants: int,
    new_n_regimes: int,
    new_n_variants: int,
) -> None:
    """Fail fast on invalid augmentation settings.

    Two expansion modes are supported (production shapes; smaller shapes are
    allowed so integration tests can exercise the same code paths):
      - +1 regime / +1 variant rectangular expansion (7x7 -> 8x8);
      - variant-only expansion, regimes fixed (7xV -> 7x(V+1)).
    """
    if cfg.gvendi_selection_method not in SELECTION_METHODS:
        raise ValueError(
            f"gvendi_selection_method must be one of {SELECTION_METHODS}, "
            f"got {cfg.gvendi_selection_method!r}"
        )
    if cfg.gvendi_gradient_target != "adapter":
        raise ValueError(
            "only gvendi_gradient_target='adapter' is supported in the first "
            f"version, got {cfg.gvendi_gradient_target!r}"
        )
    variant_only = new_n_regimes == prev_n_regimes
    if new_n_variants != prev_n_variants + 1 or (
        not variant_only and new_n_regimes != prev_n_regimes + 1
    ):
        raise ValueError(
            "gvendi only supports the +1 regime / +1 variant rectangular "
            "expansion or the variant-only (+1 variant, regimes fixed) "
            f"expansion; got {prev_n_regimes}x{prev_n_variants} -> "
            f"{new_n_regimes}x{new_n_variants}"
        )
    if not variant_only:
        new_regime_index = cfg.gvendi_new_regime_index
        if new_regime_index == -1:
            new_regime_index = prev_n_regimes
        if new_regime_index != prev_n_regimes:
            raise ValueError(
                f"gvendi_new_regime_index must be {prev_n_regimes} (the next "
                f"free regime index) or -1, got {cfg.gvendi_new_regime_index}"
            )
    if cfg.gvendi_projection_dim <= 0:
        raise ValueError("gvendi_projection_dim must be positive")
    if cfg.gvendi_probe_batches <= 0:
        raise ValueError("gvendi_probe_batches must be positive")
    for name in ("gvendi_mase_low_quantile", "gvendi_mase_high_quantile",
                 "gvendi_sparse_cluster_fraction", "gvendi_centroid_distance_quantile"):
        v = getattr(cfg, name)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must lie in [0, 1], got {v}")
    if cfg.gvendi_mase_low_quantile >= cfg.gvendi_mase_high_quantile:
        raise ValueError("gvendi_mase_low_quantile must be < gvendi_mase_high_quantile")
    if cfg.gvendi_candidates_per_existing_regime < 1:
        raise ValueError("need >= 1 candidate per existing regime")
    if not variant_only and cfg.gvendi_candidates_new_regime < new_n_variants:
        raise ValueError(
            f"gvendi_candidates_new_regime={cfg.gvendi_candidates_new_regime} "
            f"is below the new-regime quota of {new_n_variants}"
        )
    frac_sum = cfg.gvendi_frac_perturb + cfg.gvendi_frac_interpolate + cfg.gvendi_frac_explore
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(f"candidate strategy fractions must sum to 1, got {frac_sum}")
    if not (0.0 < cfg.gvendi_new_client_batch_fraction <= 1.0):
        raise ValueError("gvendi_new_client_batch_fraction must lie in (0, 1]")
    if cfg.gvendi_new_client_sampling_multiplier < 1.0:
        raise ValueError("gvendi_new_client_sampling_multiplier must be >= 1")
    if cfg.gvendi_forgetting_tolerance < 0.0:
        raise ValueError("gvendi_forgetting_tolerance must be >= 0")
