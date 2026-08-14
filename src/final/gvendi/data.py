"""Manifest-driven synthetic client construction.

The legacy pipeline (synthetic_data.build_synthetic_client_series) regenerates
the whole grid at every stage shape, which changes the *existing* clients'
generator parameters when n_regimes grows (their parameter centers depend on
regime_id / (n_regimes - 1)). For gvendi the 49 active clients must stay the
exact clients that were probed, so this module rebuilds them with their
7x7-stage parameters and RNG streams (bit-identical data), and builds the 15
selected clients from the generator parameters recorded in the selection
manifest.

Nothing in synthetic_data.py is modified; its per-client generation loop is
reproduced here (same RNG construction and consumption order).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_generation")
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from synthetic_data import (  # noqa: E402
    SyntheticWindowDataset,
    _generate_series_for_family,
    _make_regime_params,
    _moving_average_smooth,
    _regime_family,
    _sanitize_series,
    compute_mase_denom_from_train_series,
    compute_seasonal_naive_mape_from_dataset,
    estimate_client_feature_vector_from_series_list,
    get_descriptor_feature_names,
    sample_kernel_synth_series,
    split_series_gift_style,
    tsmixup,
)

REGIME_FAMILIES = [
    "trend_seasonal",
    "sarima_like",
    "long_memory",
    "smooth_seasonal",
    "low_frequency",
    "multiplicative_seasonal",
    "high_frequency",
]

# Offset separating candidate data-RNG streams from the training grid's
# streams (grid series seeds live in [seed, seed + n_clients*100 + n_series]).
_CANDIDATE_SEED_OFFSET = 900_000


def candidate_data_seed(base_seed: int, candidate_index: int) -> int:
    """Deterministic, collision-free RNG seed base for candidate index *i*."""
    return int(base_seed) + _CANDIDATE_SEED_OFFSET + int(candidate_index) * 1_000


def compute_total_len(cfg, seq_len: int) -> int:
    """Series length used by the legacy generator (build_synthetic_client_series)."""
    total_len = int(getattr(cfg, "synthetic_series_length", 4000))
    long_horizon_factor = int(getattr(cfg, "synthetic_long_horizon_factor", 4))
    context_margin = int(getattr(cfg, "synthetic_context_margin", 128))
    min_needed = max(
        seq_len + cfg.horizon + context_margin,
        2 * (seq_len + cfg.horizon),
        long_horizon_factor * (seq_len + cfg.horizon),
    )
    return max(total_len, min_needed)


def make_grid_params(cfg, regime_id: int, variant_id: int, n_regimes: int) -> Dict:
    """Generator parameters for a legacy grid cell at the given grid shape."""
    return _make_regime_params(
        regime_id,
        variant_id,
        n_regimes,
        cfg.seed,
        getattr(cfg, "synthetic_config_variant", "baseline"),
        separability=str(
            getattr(cfg, "synthetic_regime_separability", "medium")
        ).lower(),
    )


def make_new_regime_params(
    cfg,
    new_regime_index: int,
    candidate_index: int,
    n_regimes_new: int,
    family_override: str = "",
    seed: int | None = None,
) -> Dict:
    """Sample generator parameters for the fixed new regime's family.

    With a family override, parameters are drawn using a base regime id whose
    legacy family matches the requested one (the family branch in
    _make_regime_params keys off regime_id), then relabelled to the new
    regime index by the caller's metadata.
    """
    seed = cfg.gvendi_candidate_seed if seed is None else seed
    regime_for_params = new_regime_index
    if family_override:
        if family_override not in REGIME_FAMILIES:
            raise ValueError(
                f"unknown regime family {family_override!r}; "
                f"expected one of {REGIME_FAMILIES}"
            )
        base = REGIME_FAMILIES.index(family_override)
        # Keep the family cycle aligned: pick the largest regime id <= new
        # index whose family matches, falling back to the family index itself.
        regime_for_params = base
        for r in range(new_regime_index, -1, -1):
            if _regime_family(r) == family_override:
                regime_for_params = r
                break
    return _make_regime_params(
        regime_for_params,
        candidate_index,
        n_regimes_new,
        seed,
        getattr(cfg, "synthetic_config_variant", "baseline"),
        separability=str(
            getattr(cfg, "synthetic_regime_separability", "medium")
        ).lower(),
    )


# ---------------------------------------------------------------------------
# Single-client series generation (mirrors build_synthetic_client_series)
# ---------------------------------------------------------------------------

def build_one_client_series(
    cfg,
    params: Dict,
    rng_seed: int,
    series_seed_base: int,
    total_len: int,
    n_series: int,
    n_gp: int,
    n_mixup: int,
) -> tuple[List[np.ndarray], Dict[str, float]]:
    """Generate one client's raw series list.

    RNG construction and consumption order match the legacy per-client loop
    exactly, so calling this with the legacy seeds (rng_seed = seed + 1000 +
    client_id, series_seed_base = seed + client_id * 100, legacy counts)
    reproduces that client's data bit-for-bit.
    """
    rng = np.random.default_rng(rng_seed)

    base_series = []
    for k in range(n_series):
        srng = np.random.default_rng(series_seed_base + k)
        s = _generate_series_for_family(srng, total_len, params)
        base_series.append(_sanitize_series(s))

    ks_feat_accum = {
        "kernel_linear": 0.0,
        "kernel_rbf": 0.0,
        "kernel_periodic": 0.0,
        "kernel_noise": 0.0,
    }

    if getattr(cfg, "synthetic_use_kernel_synth", True) and n_gp > 0:
        gp_series = []
        gp_weight = float(getattr(cfg, "synthetic_kernel_blend_weight", 0.35))
        for _ in range(n_gp):
            samp, feat = sample_kernel_synth_series(
                rng,
                total_len,
                int(getattr(cfg, "synthetic_kernel_terms_min", 1)),
                int(getattr(cfg, "synthetic_kernel_terms_max", 3)),
                float(params["season_period_1"]),
            )
            if params["family"] in {"smooth_seasonal", "low_frequency"}:
                width = int(max(3, round(params.get("downsample_factor", 1.0) + 2)))
                samp = _moving_average_smooth(samp, width)
            blended = (
                (1.0 - gp_weight) * samp
                + gp_weight * base_series[int(rng.integers(0, len(base_series)))]
            )
            gp_series.append(_sanitize_series(blended))
            for k, v in feat.items():
                ks_feat_accum[k] += float(v)
        base_series.extend(gp_series)

    if getattr(cfg, "synthetic_use_mixup", True) and n_mixup > 0:
        base_series.extend(
            tsmixup(
                rng,
                base_series,
                n_mixup,
                int(getattr(cfg, "synthetic_max_mix_components", 3)),
                float(getattr(cfg, "synthetic_mixup_alpha", 0.7)),
            )
        )

    return base_series, ks_feat_accum


def make_client_datasets_from_series(
    cfg,
    base_series: List[np.ndarray],
    seq_len: int,
    batch_size: int,
) -> Dict | None:
    """Split series and build the train/val/test loader dict for one client.

    Mirrors the per-client portion of make_synthetic_clients. Returns None
    when no series survives splitting (caller decides how to handle it).
    """
    train_series, val_series, test_series = [], [], []
    for s in base_series:
        tr, va, te = split_series_gift_style(
            s, seq_len=seq_len, horizon=cfg.horizon, test_frac=cfg.test_frac
        )
        if tr is None:
            continue
        train_series.append(tr)
        val_series.append(va)
        test_series.append(te)

    ds_kwargs = dict(
        seq_len=seq_len,
        horizon=cfg.horizon,
        normalize_per_series=cfg.normalize_per_series,
        normalization_eps=cfg.normalization_eps,
        clip_scale_min=cfg.clip_scale_min,
    )
    train_ds = SyntheticWindowDataset(train_series, **ds_kwargs)
    val_ds = SyntheticWindowDataset(val_series, **ds_kwargs)
    test_ds = SyntheticWindowDataset(test_series, **ds_kwargs)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        return None

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers
        ),
        "train_series": train_series,
        "train_dataset": train_ds,
        "mase_denom": compute_mase_denom_from_train_series(
            train_series, seasonality=getattr(cfg, "mase_seasonality", 1)
        ),
        "seasonal_naive_mape": compute_seasonal_naive_mape_from_dataset(
            test_ds,
            seasonality=getattr(cfg, "mase_seasonality", 1),
            max_batches=cfg.eval_batches,
            batch_size=batch_size,
        ),
    }


def compute_client_feature_vector(cfg, base_series: List[np.ndarray], seq_len: int) -> torch.Tensor:
    """Raw (unnormalised) descriptor feature vector for one client."""
    descriptor_set = str(getattr(cfg, "descriptor_set", "basic")).lower()
    return estimate_client_feature_vector_from_series_list(
        base_series,
        seq_len=seq_len,
        max_windows_per_series=int(getattr(cfg, "client_feature_windows_per_series", 8)),
        descriptor_set=descriptor_set,
    )


def build_probe_client(
    cfg,
    params: Dict,
    data_seed: int,
    seq_len: int,
    batch_size: int,
) -> Dict | None:
    """Probe-scale client: reduced data volume for gradients/metrics/features.

    The RNG streams are a deterministic function of *data_seed*, and the
    probe series are a prefix of the series generated by the full build for
    the same seed (same per-series seeds, fewer of them).
    """
    total_len = compute_total_len(cfg, seq_len)
    series, ks_feat = build_one_client_series(
        cfg,
        params,
        rng_seed=data_seed + 1000,
        series_seed_base=data_seed,
        total_len=total_len,
        n_series=int(cfg.gvendi_probe_series_per_client),
        n_gp=int(cfg.gvendi_probe_gp_samples),
        n_mixup=int(cfg.gvendi_probe_mixup_per_client),
    )
    bundle = make_client_datasets_from_series(cfg, series, seq_len, batch_size)
    if bundle is None:
        return None
    bundle["raw_feature_vector"] = compute_client_feature_vector(cfg, series, seq_len)
    bundle["ks_feat_accum"] = ks_feat
    bundle["series"] = series
    return bundle


# ---------------------------------------------------------------------------
# Full next-stage client set from a selection manifest
# ---------------------------------------------------------------------------

def build_stage_clients_from_manifest(
    cfg,
    selected_clients: List[Dict],
    prev_n_regimes: int,
    prev_n_variants: int,
    new_n_regimes: int,
    new_n_variants: int,
    seq_len: int,
    batch_size: int,
):
    """Build the full next-stage client set (e.g. 64 clients for 8x8).

    Grid-major ordering (regime-major, variant within), matching the legacy
    client_id convention:
      - cells (r < prev_R, v < prev_V): the original clients, regenerated with
        their previous-stage parameters and RNG streams (bit-identical data);
      - cells (r < prev_R, v == prev_V): the selected candidate for regime r;
      - cells (r == prev_R, *): the selected new-regime candidates by
        variant_index.

    Returns (clients, meta_rows, normalized_client_features, feature_stats).
    """
    if bool(getattr(cfg, "use_oracle_regime_id", False)):
        raise NotImplementedError(
            "gvendi manifest-driven generation does not support "
            "use_oracle_regime_id features"
        )

    selected_by_cell = {
        (int(row["regime_index"]), int(row["variant_index"])): row
        for row in selected_clients
    }

    total_len = compute_total_len(cfg, seq_len)
    descriptor_set = str(getattr(cfg, "descriptor_set", "basic")).lower()
    separability = str(
        getattr(cfg, "synthetic_regime_separability", "medium")
    ).lower()
    n_series = int(getattr(cfg, "synthetic_series_per_client", 10))
    n_gp = int(getattr(cfg, "synthetic_gp_samples_per_client", 16))
    n_mixup = int(getattr(cfg, "synthetic_mixup_per_client", 8))

    clients, meta_rows, feature_rows = [], [], []
    client_id = 0
    for regime_id in range(new_n_regimes):
        for variant_id in range(new_n_variants):
            is_original = regime_id < prev_n_regimes and variant_id < prev_n_variants
            if is_original:
                old_client_id = regime_id * prev_n_variants + variant_id
                params = make_grid_params(cfg, regime_id, variant_id, prev_n_regimes)
                series, ks_feat = build_one_client_series(
                    cfg,
                    params,
                    rng_seed=cfg.seed + 1000 + old_client_id,
                    series_seed_base=cfg.seed + old_client_id * 100,
                    total_len=total_len,
                    n_series=n_series,
                    n_gp=n_gp,
                    n_mixup=n_mixup,
                )
                source_candidate_id = None
            else:
                cell = (regime_id, variant_id)
                if cell not in selected_by_cell:
                    raise ValueError(
                        f"selection manifest has no client for grid cell {cell}"
                    )
                row = selected_by_cell[cell]
                params = dict(row["generator_parameters"])
                data_seed = int(row["data_seed"])
                series, ks_feat = build_one_client_series(
                    cfg,
                    params,
                    rng_seed=data_seed + 1000,
                    series_seed_base=data_seed,
                    total_len=total_len,
                    n_series=n_series,
                    n_gp=n_gp,
                    n_mixup=n_mixup,
                )
                source_candidate_id = row["candidate_id"]

            bundle = make_client_datasets_from_series(cfg, series, seq_len, batch_size)
            if bundle is None:
                raise RuntimeError(
                    f"client regime_{regime_id:02d}_variant_{variant_id:02d} "
                    f"produced no valid train/val/test windows"
                )
            feat_vec = compute_client_feature_vector(cfg, series, seq_len)

            meta = {
                "client_id": client_id,
                "regime_id": regime_id,
                "variant_id": variant_id,
                "dataset": "synthetic",
                "regime": f"regime_{regime_id:02d}",
                "regime_variant": f"regime_{regime_id:02d}_variant_{variant_id:02d}",
                "freq": "synthetic",
                "family": params["family"],
                "descriptor_set": descriptor_set,
                "synthetic_regime_separability": separability,
                "use_oracle_regime_id": False,
                **params,
                **ks_feat,
                "n_raw_series": len(series),
                "is_new_client": not is_original,
                "source_candidate_id": source_candidate_id,
            }
            bundle.pop("series", None)
            clients.append(bundle)
            meta_rows.append(meta)
            feature_rows.append(feat_vec)
            client_id += 1

    cfg.synthetic_feature_names = tuple(get_descriptor_feature_names(descriptor_set))

    client_features = torch.stack(feature_rows, dim=0).float()
    if getattr(cfg, "normalize_client_features", True):
        mu = client_features.mean(dim=0, keepdim=True)
        sd = client_features.std(dim=0, keepdim=True).clamp_min(1e-6)
        client_features = (client_features - mu) / sd
    else:
        mu = torch.zeros(1, client_features.shape[1])
        sd = torch.ones(1, client_features.shape[1])
    feature_stats = {"mean": mu.squeeze(0).clone(), "std": sd.squeeze(0).clone()}
    return clients, meta_rows, client_features, feature_stats
