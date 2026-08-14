"""ReLoRA-specific configuration, extending the base federated config."""

from __future__ import annotations

import sys
import os

# Ensure the base config package is importable
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from dataclasses import dataclass
from config import Config  # noqa: E402  (base config from synthetic_fed_hnet_lora)


@dataclass
class ReLoRAConfig(Config):
    # ------------------------------------------------------------------
    # Curriculum structure
    # ------------------------------------------------------------------
    relora_start_regimes: int = 5
    relora_start_variants: int = 5
    relora_num_rounds: int = 3

    # Hard ceiling on communication rounds within a single curriculum round.
    # Prevents a run from never finishing if patience is never triggered.
    relora_max_rounds_per_curriculum: int = 200

    # Optional explicit per-stage round-cap schedule, parallel to
    # relora_curriculum_schedule: "100,50,100,50,100,50" gives curriculum round 0
    # a 100-round ceiling, round 1 a 50-round ceiling, etc. Must have the same
    # number of comma-separated entries as relora_curriculum_schedule. Empty
    # string means every curriculum round uses the flat
    # relora_max_rounds_per_curriculum ceiling instead.
    relora_max_rounds_schedule: str = ""

    # ------------------------------------------------------------------
    # Per-client patience (replaces server-level patience)
    # ------------------------------------------------------------------
    relora_client_patience: int = 10

    # Minimum absolute improvement in validation loss to reset a client counter.
    relora_patience_min_delta: float = 1e-4

    # "all": every full-eval round updates every client's patience counter,
    # so patience = consecutive evaluations without improvement, uniform
    # across clients and independent of client sampling.
    # "sampled": legacy behavior — a client's counter only advances on rounds
    # it is sampled, so with C clients and S sampled per round a stage must
    # run ~patience * C / S rounds before patience can possibly trigger.
    relora_patience_scope: str = "all"

    # ------------------------------------------------------------------
    # Stage-best checkpoint selection
    # ------------------------------------------------------------------
    # At the end of each curriculum stage, restore the server (hnet + emb)
    # and global head to the evaluated round with the lowest mean client
    # validation MAE, instead of carrying the last round's state into the
    # boundary / final checkpoint.
    relora_restore_best: bool = True

    # ------------------------------------------------------------------
    # Adam moment pruning at round boundaries
    # ------------------------------------------------------------------
    relora_adam_pruning_pct: float = 0.9

    # Pruning strategy at round boundaries: "magnitude" (default, see
    # adam_reset.prune_adam_moments), "variance" (see
    # adam_reset.partial_adam_reset, driven by GradVarianceTracker), or
    # "full" (see adam_reset.full_adam_reset — wipes 100% of optimizer state).
    relora_pruning_method: str = "magnitude"

    # When relora_pruning_method == "variance": use SNR (mean^2 / variance)
    # as the importance score instead of raw variance.
    relora_use_snr: bool = False

    # Fraction of Adam moments retained (by importance) when using the
    # "variance" pruning method. Analogous to 1 - relora_adam_pruning_pct.
    relora_retain_fraction: float = 0.10

    # ------------------------------------------------------------------
    # LR schedule across curriculum rounds
    # ------------------------------------------------------------------
    # One warmup step == one comm round, and the counter restarts at every
    # stage boundary — keep this well below the shortest stage cap or the
    # server spends entire stages on the LR ramp and never trains at target LR.
    relora_lr_warmup_steps: int = 20

    # Multiply server_lr by this factor each curriculum round.
    # 1.0 keeps it constant; 0.9 gives a 10% decay per round.
    relora_lr_decay_per_round: float = 1.0

    # Within-stage server LR schedule applied after warmup: "constant" holds
    # the stage target LR; "cosine" decays it to
    # relora_lr_min_factor * target_lr by the stage's round cap.
    relora_lr_schedule: str = "constant"
    relora_lr_min_factor: float = 0.1

    # Optional explicit curriculum schedule, overrides start_regimes/start_variants
    # and relora_num_rounds. Format: "NxM,NxM,..." e.g. "5x5,5x5,6x6,6x6,7x7,7x7".
    # Empty string means use the default +k expansion formula.
    relora_curriculum_schedule: str = ""
