"""FedChronos-style clip-and-noise perturbation of the transmitted LoRA
update (see the gvendi DP proposal). Off by default and a strict no-op
when off; see train_8x8.run_gvendi_stage for the call site.

This is NOT full-pipeline client-level differential privacy: only the
LoRA adapter update is clipped and noised here. The forecast head is
still federated-averaged unmodified (train_8x8._average_state_dicts on
local_head_states), so there is no end-to-end (epsilon, delta)
guarantee for a client's full contribution to a round. gvendi_dp_epsilon
is a FedChronos-matched per-round noise-calibration parameter, not a
composed privacy budget -- there is no accountant tracking cumulative
loss across communication rounds.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

DP_MODES = ("off", "clip_only", "noise_only", "dp")

_DP_RNG_STATE_FILENAME = "dp_rng_state.pt"


def sigma_from_epsilon(epsilon: float, delta: float) -> float:
    """FedChronos-matched per-round noise multiplier -- see module
    docstring for why this is not a rigorous end-to-end privacy budget
    for a gvendi run."""
    return math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def resolve_dp_params(cfg) -> tuple[Optional[float], float]:
    """Map cfg.gvendi_dp_mode to (clip_norm, noise_std) for
    clip_and_noise_delta.

    clip_norm is None when clipping is off (mode in {"off","noise_only"}).
    noise_std is the absolute Gaussian std added, sigma * C, shared
    between "noise_only" and "dp" so that comparing the two isolates the
    effect of clipping alone, holding the noise distribution fixed.
    """
    mode = cfg.gvendi_dp_mode
    if mode not in DP_MODES:
        raise ValueError(f"gvendi_dp_mode must be one of {DP_MODES}, got {mode!r}")

    clip_norm = cfg.gvendi_dp_clip_norm if mode in ("clip_only", "dp") else None
    if mode in ("noise_only", "dp"):
        sigma = sigma_from_epsilon(cfg.gvendi_dp_epsilon, cfg.gvendi_dp_delta)
        noise_std = sigma * cfg.gvendi_dp_clip_norm
    else:
        noise_std = 0.0
    return clip_norm, noise_std


def clip_and_noise_delta(
    delta: torch.Tensor,
    clip_norm: Optional[float],
    noise_std: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict]:
    """Clip `delta` to an L2 ball of radius clip_norm (skipped if None),
    then add N(0, noise_std^2 I) (skipped if noise_std <= 0).

    `delta` and `generator` must both be CPU-resident. This is not a new
    constraint on callers: local_train_lora_and_head_steps already
    returns a .cpu() tensor (synthetic_fed_hnet_lora/client.py:240), as
    does server.generate_lora_flat(...).detach().cpu() at the gvendi call
    site (train_8x8.run_gvendi_stage). Asserting it here fails loudly if
    that guarantee is ever broken, instead of silently mixing CPU/CUDA
    generator state.

    Returns (noised_delta, diagnostics) where diagnostics has raw_norm,
    clipped_norm, noise_norm, noised_norm, and noise_signal_ratio
    (= noise_norm / (clipped_norm + 1e-12)).
    """
    assert delta.device.type == "cpu", "clip_and_noise_delta expects a CPU tensor"

    raw_norm = delta.norm().item()
    if clip_norm is not None:
        scale = min(1.0, clip_norm / (raw_norm + 1e-12))
        clipped = delta * scale
    else:
        clipped = delta
    clipped_norm = clipped.norm().item()

    if noise_std > 0.0:
        noise = torch.randn(
            delta.shape, generator=generator, dtype=delta.dtype
        ) * noise_std
    else:
        noise = torch.zeros_like(delta)
    noise_norm = noise.norm().item()

    noised = clipped + noise
    diagnostics = {
        "raw_norm": raw_norm,
        "clipped_norm": clipped_norm,
        "noise_norm": noise_norm,
        "noised_norm": noised.norm().item(),
        "noise_signal_ratio": noise_norm / (clipped_norm + 1e-12),
    }
    return noised, diagnostics


# ---------------------------------------------------------------------------
# Per-client generator lifecycle across curriculum stages.
#
# Each curriculum stage is a separate process invocation (run_gvendi_8x8,
# run_gvendi_variant_expansion, run_gvendi_regime_expansion): stage t+1
# restores hnet/embedding weights from server.pt and the forecast head
# from global_head_state.pt, both loaded from the previous stage's
# checkpoint dir, and at the end of its own run saves the same two files
# for the next stage to load. The DP noise generators are checkpointed
# through the exact same chain (as dp_rng_state.pt) so each client's noise
# stream is a true continuation across the whole pipeline, rather than
# restarting identically at every stage boundary.
# ---------------------------------------------------------------------------


def load_or_init_generators(
    cfg,
    prev_ckpt_dir: Optional[str],
    n_clients: Optional[int] = None,
) -> dict:
    """Restore per-client DP generators from the previous stage's
    checkpoint dir if a dp_rng_state.pt is present there, else seed fresh
    ones from cfg.gvendi_dp_seed. Clients with no prior state (either
    because this is the first DP-enabled stage of a chain, or because
    they were just introduced by curriculum expansion) are seeded fresh
    the first time they are sampled -- there is nothing to continue.
    """
    n_clients = cfg.n_clients if n_clients is None else n_clients
    saved_state = {}
    if prev_ckpt_dir is not None:
        state_path = os.path.join(prev_ckpt_dir, _DP_RNG_STATE_FILENAME)
        if os.path.exists(state_path):
            saved_state = torch.load(state_path, map_location="cpu", weights_only=False)

    generators = {}
    for cid in range(n_clients):
        gen = torch.Generator(device="cpu")
        if cid in saved_state:
            gen.set_state(saved_state[cid])
        else:
            gen.manual_seed(cfg.gvendi_dp_seed * 100_003 + cid)
        generators[cid] = gen
    return generators


def save_generator_state(generators: dict, ckpt_dir: str) -> None:
    state = {cid: gen.get_state() for cid, gen in generators.items()}
    torch.save(state, os.path.join(ckpt_dir, _DP_RNG_STATE_FILENAME))
