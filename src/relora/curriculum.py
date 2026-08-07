"""Curriculum scheduling, embedding expansion, and round-boundary orchestration.

Responsibilities
----------------
- CurriculumScheduler.get_round_shape(k)   → (n_regimes, n_variants) for round k
- CurriculumScheduler.save_checkpoint(...)  → persist server + head + opt state
- CurriculumScheduler.execute_boundary(...)  → Adam reset, embedding expansion, LR=0
- CurriculumScheduler.apply_warmup(...)     → linear LR ramp
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "synthetic_fed_hnet_lora")
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from .adam_reset import full_adam_reset, partial_adam_reset, prune_adam_moments  # noqa: E402
from .config import ReLoRAConfig  # noqa: E402
from .grad_variance import GradVarianceTracker  # noqa: E402


# ---------------------------------------------------------------------------
# Semantic key helpers (regime/variant → embedding-table index)
# ---------------------------------------------------------------------------

def build_key_to_index(meta_rows: list[dict]) -> dict[str, int]:
    """Map "regime_XX_variant_YY" → embedding-table position."""
    return {
        f"regime_{row['regime_id']:02d}_variant_{row['variant_id']:02d}": i
        for i, row in enumerate(meta_rows)
    }


def _compute_regime_mean_embeddings(
    emb_weight: torch.Tensor,
    old_key_to_index: dict[str, int],
    old_n_regimes: int,
    old_n_variants: int,
) -> dict[int, torch.Tensor]:
    """Return a dict mapping regime_id → mean embedding vector over its variants."""
    means: dict[int, torch.Tensor] = {}
    for r in range(old_n_regimes):
        vecs = []
        for v in range(old_n_variants):
            key = f"regime_{r:02d}_variant_{v:02d}"
            if key in old_key_to_index:
                idx = old_key_to_index[key]
                vecs.append(emb_weight[idx].detach().cpu())
        if vecs:
            means[r] = torch.stack(vecs).mean(0)
    return means


# ---------------------------------------------------------------------------
# Embedding expansion
# ---------------------------------------------------------------------------

def expand_server_embeddings(
    server,
    old_meta_rows: list[dict],
    new_meta_rows: list[dict],
    old_n_regimes: int,
    old_n_variants: int,
) -> None:
    """Expand server.emb and server.client_features in-place for the new client set.

    New client embeddings are initialised from the mean of the closest existing
    regime (by index), not from Gaussian noise.  server.client_features is set
    to the new round's pre-computed features (passed in via new_meta_rows –
    the caller is expected to have already updated server.client_features to
    the new-round tensor before calling this function if fixed features are used).

    After this call server.emb has ``len(new_meta_rows)`` rows and the old
    embedding parameter object is replaced.  The optimizer must be updated by
    the caller (old embedding params removed, new ones added).
    """
    device = server.device
    emb_dim = server.emb.weight.shape[1]

    old_weight = server.emb.weight.data  # [n_old, emb_dim]
    old_key_to_index = build_key_to_index(old_meta_rows)
    new_key_to_index = build_key_to_index(new_meta_rows)

    regime_means = _compute_regime_mean_embeddings(
        old_weight, old_key_to_index, old_n_regimes, old_n_variants
    )

    n_new = len(new_meta_rows)
    new_weight = torch.zeros(n_new, emb_dim)

    n_preserved, n_regime_init, n_fresh = 0, 0, 0
    for row in new_meta_rows:
        key = f"regime_{row['regime_id']:02d}_variant_{row['variant_id']:02d}"
        new_idx = new_key_to_index[key]

        if key in old_key_to_index:
            # Carry over existing embedding unchanged.
            new_weight[new_idx] = old_weight[old_key_to_index[key]].detach().cpu()
            n_preserved += 1
        else:
            # New client: initialise from mean of closest existing regime.
            regime_id = int(row["regime_id"])
            # Find the closest existing regime by index (fallback to 0).
            closest = min(regime_means.keys(), key=lambda r: abs(r - regime_id), default=None)
            if closest is not None:
                new_weight[new_idx] = regime_means[closest].clone()
                n_regime_init += 1
            else:
                nn.init.normal_(new_weight[new_idx : new_idx + 1], mean=0.0, std=0.02)
                n_fresh += 1

    new_emb = nn.Embedding(n_new, emb_dim).to(device)
    new_emb.weight.data.copy_(new_weight.to(device))
    server.emb = new_emb

    print(
        f"[curriculum] embedding expanded {len(old_meta_rows)} → {n_new} clients "
        f"(preserved={n_preserved}, regime-mean-init={n_regime_init}, "
        f"fresh-normal-init={n_fresh})"
    )


def _rebuild_optimizer_with_new_emb(
    server,
    old_emb_param: nn.Parameter,
    weight_decay: float = 1e-4,
) -> None:
    """Swap the old embedding parameter out of the optimizer and add the new one.

    Hypernetwork parameter moments are already pruned in-place; we preserve them
    by keeping the same param objects in the optimizer.  Only the embedding entry
    is replaced (old moments are discarded; new embedding starts fresh).
    """
    opt = server.opt

    # Remove old embedding from state and param groups.
    if old_emb_param in opt.state:
        del opt.state[old_emb_param]
    for group in opt.param_groups:
        group["params"] = [p for p in group["params"] if p is not old_emb_param]

    # Add new embedding with LR already 0 (set by prune_adam_moments).
    if server.learnable_embeddings:
        opt.add_param_group(
            {
                "params": list(server.emb.parameters()),
                "lr": 0.0,
                "weight_decay": weight_decay,
            }
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Orchestrates round boundaries for the ReLoRA curriculum.

    Parameters
    ----------
    cfg:
        A ReLoRAConfig instance.
    """

    def __init__(self, cfg: ReLoRAConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Shape queries
    # ------------------------------------------------------------------

    def _parse_schedule(self) -> list[tuple[int, int]]:
        """Parse relora_curriculum_schedule into a list of (n_regimes, n_variants)."""
        raw = getattr(self.cfg, "relora_curriculum_schedule", "").strip()
        if not raw:
            return []
        pairs = []
        for token in raw.split(","):
            token = token.strip()
            if "x" not in token:
                raise ValueError(f"Invalid schedule token {token!r}; expected 'NxM'")
            r, v = token.split("x", 1)
            pairs.append((int(r), int(v)))
        return pairs

    def get_round_shape(self, k: int) -> tuple[int, int]:
        """Return (n_regimes, n_variants) for curriculum round *k*."""
        schedule = self._parse_schedule()
        if schedule:
            return schedule[k]
        return (
            self.cfg.relora_start_regimes + k,
            self.cfg.relora_start_variants + k,
        )

    def num_rounds(self) -> int:
        """Total curriculum rounds — from schedule length or relora_num_rounds."""
        schedule = self._parse_schedule()
        return len(schedule) if schedule else self.cfg.relora_num_rounds

    def _parse_max_rounds_schedule(self) -> list[int]:
        """Parse relora_max_rounds_schedule into a list of per-stage round caps."""
        raw = getattr(self.cfg, "relora_max_rounds_schedule", "").strip()
        if not raw:
            return []
        caps = [int(tok.strip()) for tok in raw.split(",")]
        schedule = self._parse_schedule()
        if schedule and len(caps) != len(schedule):
            raise ValueError(
                f"relora_max_rounds_schedule has {len(caps)} entries but "
                f"relora_curriculum_schedule has {len(schedule)}; they must match"
            )
        return caps

    def get_max_rounds(self, k: int) -> int:
        """Return the communication-round ceiling for curriculum round *k*.

        Falls back to the flat relora_max_rounds_per_curriculum ceiling when
        relora_max_rounds_schedule is empty.
        """
        caps = self._parse_max_rounds_schedule()
        if caps:
            return caps[k]
        return self.cfg.relora_max_rounds_per_curriculum

    def get_target_lr(self, k: int) -> float:
        """Server learning rate for curriculum round *k* (with per-round decay)."""
        return self.cfg.server_lr * (self.cfg.relora_lr_decay_per_round ** k)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        server,
        global_head_state: dict,
        out_dir: str,
        curriculum_round: int,
    ) -> str:
        """Persist the end-of-round state to *out_dir/_relora_round_k_ckpt/*.

        Saves: server.pt, global_head_state.pt, optimizer.pt.
        Returns the checkpoint directory path.
        """
        ckpt_dir = os.path.join(out_dir, f"_relora_round_{curriculum_round}_ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        server.save(ckpt_dir)
        torch.save(global_head_state, os.path.join(ckpt_dir, "global_head_state.pt"))
        torch.save(server.opt.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        print(f"[curriculum] round {curriculum_round} checkpoint → {ckpt_dir}")
        return ckpt_dir

    # ------------------------------------------------------------------
    # Round boundary
    # ------------------------------------------------------------------

    def execute_boundary(
        self,
        server,
        old_meta_rows: list[dict],
        new_meta_rows: list[dict],
        new_client_features: torch.Tensor | None,
        old_n_regimes: int,
        old_n_variants: int,
        weight_decay: float = 1e-4,
        grad_tracker: GradVarianceTracker | None = None,
    ) -> None:
        """Execute all round-boundary actions (call AFTER saving checkpoint).

        Steps (in order):
        1. Reset Adam moments for hnet + existing embedding. Magnitude-based
           by default (relora_pruning_method == "magnitude"); if
           relora_pruning_method == "variance", uses *grad_tracker*'s
           variance (or SNR, if relora_use_snr) as the importance score via
           partial_adam_reset instead; if relora_pruning_method == "full",
           wipes 100% of optimizer state via full_adam_reset instead of
           magnitude-thresholding (which cannot reach 100%).
        2. Set all optimizer LRs to 0.
        3. Expand embedding table (preserving old, mean-init new).
        4. Update optimizer: swap old embedding param for new one.
        5. Update server.client_features if fixed features are used.

        Learning rate warmup is NOT applied here; call apply_warmup() per step
        during the first relora_lr_warmup_steps steps of round k+1.
        """
        # Step 1: prune moments.
        if self.cfg.relora_pruning_method == "variance":
            if grad_tracker is None:
                raise ValueError(
                    "relora_pruning_method='variance' requires a grad_tracker"
                )
            importance_scores = (
                grad_tracker.snr() if self.cfg.relora_use_snr else grad_tracker.variance()
            )
            if importance_scores:
                partial_adam_reset(
                    server.opt, importance_scores, retain_fraction=self.cfg.relora_retain_fraction
                )
            else:
                # First boundary after a resume: the tracker isn't checkpointed,
                # so no gradient stats exist yet. Fall back to a magnitude
                # partial reset of equivalent strength for this one boundary.
                print(
                    "[boundary] grad tracker empty (resume); falling back to "
                    f"magnitude reset keeping top {self.cfg.relora_retain_fraction:.0%}"
                )
                prune_adam_moments(server.opt, 1.0 - self.cfg.relora_retain_fraction)
            grad_tracker.reset()
            # Step 2: LR to zero (prune_adam_moments does this itself for
            # the magnitude path; partial_adam_reset does not).
            for group in server.opt.param_groups:
                group["lr"] = 0.0
        elif self.cfg.relora_pruning_method == "full":
            # Step 1–2: wipe all optimizer state, set LR = 0.
            full_adam_reset(server.opt)
        else:
            # Step 1–2: prune moments, set LR = 0.
            prune_adam_moments(server.opt, self.cfg.relora_adam_pruning_pct)

        # Hold reference to old embedding parameter before replacement.
        old_emb_param = server.emb.weight

        # Step 3: expand embedding table in-place on server.emb.
        expand_server_embeddings(
            server, old_meta_rows, new_meta_rows, old_n_regimes, old_n_variants
        )

        # Step 4: wire new embedding into optimizer.
        _rebuild_optimizer_with_new_emb(server, old_emb_param, weight_decay)

        # Step 5: replace fixed client features.
        if new_client_features is not None:
            server.client_features = new_client_features.to(server.device)
            print(
                f"[curriculum] client_features updated: "
                f"{new_client_features.shape[0]} clients, "
                f"dim={new_client_features.shape[1]}"
            )

    # ------------------------------------------------------------------
    # LR warmup
    # ------------------------------------------------------------------

    def apply_warmup(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_step: int,
        target_lrs: list[float],
    ) -> float:
        """Linearly ramp optimizer LRs from 0 toward *target_lrs*.

        Args:
            optimizer:    The server optimizer.
            warmup_step:  0-indexed step within the warmup window.
            target_lrs:   Target LR for each param group (len == n param groups).

        Returns:
            Current scale factor in (0, 1].
        """
        warmup_steps = self.cfg.relora_lr_warmup_steps
        if warmup_steps <= 0 or warmup_step >= warmup_steps:
            scale = 1.0
        else:
            scale = (warmup_step + 1) / warmup_steps

        for group, base_lr in zip(optimizer.param_groups, target_lrs):
            group["lr"] = base_lr * scale

        return scale

    def is_in_warmup(self, warmup_step: int) -> bool:
        return warmup_step < self.cfg.relora_lr_warmup_steps

    def apply_cosine_decay(
        self,
        optimizer: torch.optim.Optimizer,
        round_in_stage: int,
        warmup_rounds: int,
        max_rounds: int,
        target_lrs: list[float],
    ) -> float:
        """Cosine-decay optimizer LRs from *target_lrs* after warmup ends.

        The decay spans the post-warmup portion of the stage, reaching
        ``relora_lr_min_factor * target_lr`` at *max_rounds* (the stage's
        round cap). Stages that end early by patience simply stop partway
        down the schedule.

        Args:
            optimizer:      The server optimizer.
            round_in_stage: 1-indexed communication round within the stage.
            warmup_rounds:  Rounds consumed by LR warmup in this stage.
            max_rounds:     The stage's communication-round cap.
            target_lrs:     Target LR for each param group.

        Returns:
            Current scale factor in [relora_lr_min_factor, 1].
        """
        min_factor = self.cfg.relora_lr_min_factor
        span = max(max_rounds - warmup_rounds, 1)
        progress = min(max(round_in_stage - warmup_rounds, 0) / span, 1.0)
        scale = min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        for group, base_lr in zip(optimizer.param_groups, target_lrs):
            group["lr"] = base_lr * scale

        return scale
