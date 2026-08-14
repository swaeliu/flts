"""Candidate pool generation and the candidate quality gate.

Candidates carry provisional ids only; final grid client keys are assigned
after selection (manifest.assign_grid_cells). All RNG streams are seeded from
gvendi_candidate_seed so the pool is identical across selection methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import torch

from .data import (
    build_probe_client,
    candidate_data_seed,
    make_grid_params,
    make_new_regime_params,
)
from .grad_vendi import RademacherProjector
from .probing import ProbeEngine, make_probe_batches, stable_probe_seed

# Keys that are structural rather than continuous generator parameters.
_NON_NUMERIC_KEYS = {"family", "family_id"}


@dataclass
class CandidateSpec:
    candidate_id: str
    regime_index: int
    generator_parameters: Dict
    generation_strategy: str  # {"perturb", "interpolate", "extrapolate", "random", "new_regime"}
    data_seed: int
    source_variants: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "regime_index": self.regime_index,
            "generator_parameters": dict(self.generator_parameters),
            "generation_strategy": self.generation_strategy,
            "data_seed": self.data_seed,
            "source_variants": list(self.source_variants),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CandidateSpec":
        return cls(
            candidate_id=d["candidate_id"],
            regime_index=int(d["regime_index"]),
            generator_parameters=dict(d["generator_parameters"]),
            generation_strategy=d["generation_strategy"],
            data_seed=int(d["data_seed"]),
            source_variants=list(d.get("source_variants", [])),
        )


@dataclass
class CandidateEvaluation:
    spec: CandidateSpec
    raw_feature_vector: torch.Tensor
    feature_vector: torch.Tensor          # normalised with active-run stats
    projected_gradients: torch.Tensor     # [B, d]
    proxy_metrics: Dict
    learnability_improvement: float | None
    filter_results: Dict
    is_eligible: bool


# ---------------------------------------------------------------------------
# Generator-parameter manipulation
# ---------------------------------------------------------------------------

def sanitize_generator_params(params: Dict) -> Dict:
    """Re-apply the legacy generator's clips and AR stability constraints.

    Mirrors the final clamping block of _make_regime_params so that perturbed
    or interpolated parameters stay within declared valid ranges.
    """
    out = dict(params)
    out["noise_scale"] = float(np.clip(out["noise_scale"], 1e-4, 5.0))
    out["heteroskedasticity"] = float(np.clip(out["heteroskedasticity"], 0.0, 2.0))
    out["exp_trend_strength"] = float(np.clip(out["exp_trend_strength"], 0.0, 0.25))
    out["season_period_1"] = float(np.clip(out["season_period_1"], 2.0, 1024.0))
    out["season_period_2"] = float(np.clip(out["season_period_2"], 2.0, 2048.0))
    out["season_amp_1"] = float(np.clip(out["season_amp_1"], 0.0, 5.0))
    out["season_amp_2"] = float(np.clip(out["season_amp_2"], 0.0, 5.0))
    out["downsample_factor"] = float(np.clip(out.get("downsample_factor", 1.0), 1.0, 48.0))

    ar = np.array(
        [out["ar_coef_1"], out["ar_coef_2"], out["ar_coef_3"]], dtype=np.float64
    )
    ar_sum = np.sum(np.abs(ar))
    if ar_sum >= 0.95:
        ar = ar * (0.95 / max(ar_sum, 1e-8))
    out["ar_coef_1"], out["ar_coef_2"], out["ar_coef_3"] = [float(a) for a in ar]

    if out.get("family") == "sarima_like":
        keys = ["ar_coef_1", "ar_coef_2", "ar_coef_3", "seasonal_ar_1", "seasonal_ar_2"]
        combined = sum(abs(out.get(k, 0.0)) for k in keys)
        if combined >= 0.60:
            scale = 0.60 / combined
            for k in keys:
                if k in out:
                    out[k] = float(out[k] * scale)
    return out


def params_are_valid(params: Dict) -> bool:
    for k, v in params.items():
        if k in _NON_NUMERIC_KEYS:
            continue
        if not math.isfinite(float(v)):
            return False
    return True


def _perturb_params(params: Dict, rng: np.random.Generator, scale: float) -> Dict:
    out = dict(params)
    for k, v in params.items():
        if k in _NON_NUMERIC_KEYS:
            continue
        v = float(v)
        sigma = (0.10 * abs(v) + 0.01) * scale
        out[k] = float(v + rng.normal(0.0, sigma))
    return sanitize_generator_params(out)


def _interpolate_params(
    a: Dict, b: Dict, rng: np.random.Generator, lam_range=(0.3, 0.7)
) -> Dict:
    lam = float(rng.uniform(*lam_range))
    out = dict(a)
    for k in a:
        if k in _NON_NUMERIC_KEYS:
            continue
        out[k] = float(lam * float(a[k]) + (1.0 - lam) * float(b.get(k, a[k])))
    return sanitize_generator_params(out)


def _extrapolate_params(a: Dict, b: Dict, rng: np.random.Generator) -> Dict:
    """Controlled extrapolation from b through a, clipped to valid ranges."""
    lam = float(rng.uniform(0.5, 1.0))
    out = dict(a)
    for k in a:
        if k in _NON_NUMERIC_KEYS:
            continue
        va, vb = float(a[k]), float(b.get(k, a[k]))
        out[k] = float(va + lam * (va - vb))
    return sanitize_generator_params(out)


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------

def _strategy_counts(n: int, cfg) -> Dict[str, int]:
    n_interp = int(round(cfg.gvendi_frac_interpolate * n))
    n_explore = int(round(cfg.gvendi_frac_explore * n))
    n_perturb = n - n_interp - n_explore
    return {"perturb": max(n_perturb, 0), "interpolate": n_interp, "explore": n_explore}


def generate_candidate_pool(
    cfg,
    prev_n_regimes: int,
    prev_n_variants: int,
    new_n_regimes: int,
    new_regime_index: int,
    anchor_weights: Dict[tuple, float] | None = None,
    extra_batch: int = 0,
    regimes: List[int] | None = None,
) -> List[CandidateSpec]:
    """Generate the oversampled candidate pool.

    Existing regimes get a perturb/interpolate/explore mixture around their
    active variants (anchors optionally weighted toward sparse-gradient
    clients via *anchor_weights*, keyed by (regime, variant)); the fixed new
    regime is sampled directly from its family's parameter distribution.

    ``extra_batch > 0`` shifts every RNG stream deterministically so quota
    top-up batches never repeat the base pool.
    """
    specs: List[CandidateSpec] = []
    batch_tag = "" if extra_batch == 0 else f"b{extra_batch}_"
    seed_shift = extra_batch * 10_889

    active_regimes = list(range(prev_n_regimes)) if regimes is None else [
        r for r in regimes if r != new_regime_index
    ]
    include_new = regimes is None or new_regime_index in regimes

    global_index = extra_batch * 100_000

    for regime in active_regimes:
        rng = np.random.default_rng(
            cfg.gvendi_candidate_seed + 97 * regime + seed_shift
        )
        variant_params = {
            v: make_grid_params(cfg, regime, v, prev_n_regimes)
            for v in range(prev_n_variants)
        }
        variants = sorted(variant_params.keys())
        weights = np.array(
            [
                1.0 + (anchor_weights or {}).get((regime, v), 0.0)
                for v in variants
            ],
            dtype=np.float64,
        )
        weights = weights / weights.sum()

        counts = _strategy_counts(cfg.gvendi_candidates_per_existing_regime, cfg)
        idx_in_regime = 0
        for strategy, count in counts.items():
            for _ in range(count):
                if strategy == "perturb":
                    v = int(rng.choice(variants, p=weights))
                    params = _perturb_params(
                        variant_params[v], rng, cfg.gvendi_perturb_scale
                    )
                    sources = [v]
                    strat = "perturb"
                elif strategy == "interpolate":
                    v1, v2 = rng.choice(variants, size=2, replace=False)
                    params = _interpolate_params(
                        variant_params[int(v1)], variant_params[int(v2)], rng
                    )
                    sources = [int(v1), int(v2)]
                    strat = "interpolate"
                else:  # explore: half fresh-random, half extrapolation
                    if rng.random() < 0.5 and prev_n_variants >= 2:
                        v1, v2 = rng.choice(variants, size=2, replace=False)
                        params = _extrapolate_params(
                            variant_params[int(v1)], variant_params[int(v2)], rng
                        )
                        sources = [int(v1), int(v2)]
                        strat = "extrapolate"
                    else:
                        fresh_variant = int(rng.integers(1_000, 1_000_000))
                        params = sanitize_generator_params(
                            make_grid_params(cfg, regime, fresh_variant, prev_n_regimes)
                        )
                        sources = []
                        strat = "random"

                specs.append(
                    CandidateSpec(
                        candidate_id=f"cand_r{regime:02d}_{batch_tag}{idx_in_regime:03d}",
                        regime_index=regime,
                        generator_parameters=params,
                        generation_strategy=strat,
                        data_seed=candidate_data_seed(cfg.seed, global_index),
                        source_variants=sources,
                    )
                )
                idx_in_regime += 1
                global_index += 1

    if include_new:
        for i in range(cfg.gvendi_candidates_new_regime):
            params = sanitize_generator_params(
                make_new_regime_params(
                    cfg,
                    new_regime_index,
                    i + seed_shift,
                    new_n_regimes,
                    family_override=cfg.gvendi_new_regime_family,
                )
            )
            specs.append(
                CandidateSpec(
                    candidate_id=f"cand_r{new_regime_index:02d}_{batch_tag}{i:03d}",
                    regime_index=new_regime_index,
                    generator_parameters=params,
                    generation_strategy="new_regime",
                    data_seed=candidate_data_seed(cfg.seed, global_index),
                )
            )
            global_index += 1

    return specs


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def _numerical_validity(spec: CandidateSpec, bundle: Dict | None) -> Dict:
    """Finite params, non-degenerate data, enough context/horizon values."""
    result = {"params_finite": params_are_valid(spec.generator_parameters)}
    if bundle is None:
        result.update(
            {"has_windows": False, "target_variance_ok": False, "series_finite": False}
        )
        result["passed"] = False
        return result

    result["has_windows"] = True
    train_series = bundle["train_series"]
    all_finite = all(np.isfinite(s).all() for s in train_series)
    variances = [float(np.var(np.asarray(s, dtype=np.float64))) for s in train_series]
    result["series_finite"] = bool(all_finite)
    result["target_variance_ok"] = bool(max(variances, default=0.0) > 1e-10)
    result["passed"] = (
        result["params_finite"]
        and result["series_finite"]
        and result["target_variance_ok"]
    )
    return result


def evaluate_candidate_quality(
    cfg,
    spec: CandidateSpec,
    engine: ProbeEngine,
    projector: RademacherProjector,
    adapter_fn: Callable[[torch.Tensor], torch.Tensor],
    base_ctor,
    base_state: Dict,
    lora_cfg: Dict,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    mase_bounds: tuple[float, float],
    existing_features_same_regime: List[torch.Tensor],
    seq_len: int,
) -> CandidateEvaluation:
    """Run one candidate through probes, proxy metrics, and every filter.

    Filter order: numerical validity -> difficulty band -> learnability ->
    feature-space duplicate rejection. The (expensive) learnability test only
    runs for candidates that passed the earlier filters. The temporarily
    adapted state is discarded; diversity is always computed from the
    pre-adaptation probe gradients.
    """
    bundle = build_probe_client(
        cfg, spec.generator_parameters, spec.data_seed, seq_len, cfg.batch_size
    )

    filters: Dict = {"numerical": _numerical_validity(spec, bundle)}

    d = projector.output_dim
    if bundle is None:
        return CandidateEvaluation(
            spec=spec,
            raw_feature_vector=torch.zeros_like(feature_mean),
            feature_vector=torch.zeros_like(feature_mean),
            projected_gradients=torch.zeros(cfg.gvendi_probe_batches, d),
            proxy_metrics={},
            learnability_improvement=None,
            filter_results=filters,
            is_eligible=False,
        )

    raw_feat = bundle["raw_feature_vector"]
    norm_feat = (raw_feat - feature_mean) / feature_std

    flat = adapter_fn(norm_feat.unsqueeze(0)).squeeze(0).detach().cpu()

    probe_batches = make_probe_batches(
        bundle["train_dataset"],
        n_batches=cfg.gvendi_probe_batches,
        batch_size=cfg.gvendi_probe_batch_size,
        seed=stable_probe_seed(cfg.gvendi_probe_seed, spec.candidate_id),
    )
    raw_grads = engine.probe_gradients(flat, probe_batches)
    projected = projector.project(raw_grads)

    zero_rows = bool((raw_grads.norm(dim=1) < 1e-12).any())
    filters["numerical"]["nonzero_gradients"] = not zero_rows
    filters["numerical"]["passed"] = filters["numerical"]["passed"] and not zero_rows

    proxy = engine.evaluate(
        flat,
        bundle["test"],
        max_batches=cfg.eval_batches,
        mase_denom=bundle["mase_denom"],
        seasonal_naive_mape=bundle["seasonal_naive_mape"],
    )
    proxy_metrics = {
        "mase": proxy.get("mase"),
        "smape": proxy["smape"],
        "mae": proxy["mae"],
        "mse": proxy["mse"],
    }

    mase = proxy.get("mase")
    lo, hi = mase_bounds
    mase_ok = mase is not None and math.isfinite(mase) and lo <= mase <= hi
    filters["difficulty"] = {"proxy_mase": mase, "low": lo, "high": hi, "passed": bool(mase_ok)}

    learnability = None
    if filters["numerical"]["passed"] and mase_ok:
        learnability = engine.learnability_improvement(
            flat,
            bundle["train"],
            probe_batches,
            base_ctor=base_ctor,
            base_state=base_state,
            local_steps=cfg.gvendi_learnability_steps,
            local_lr=cfg.local_lr,
            lora_cfg=lora_cfg,
        )
        learn_ok = learnability >= cfg.gvendi_min_learnability
    else:
        learn_ok = False
    filters["learnability"] = {
        "improvement": learnability,
        "threshold": cfg.gvendi_min_learnability,
        "passed": bool(learn_ok),
        "evaluated": learnability is not None,
    }

    tau = cfg.gvendi_min_existing_feature_distance
    if tau > 0.0 and existing_features_same_regime:
        min_dist = min(
            float((norm_feat - f).norm().item())
            for f in existing_features_same_regime
        )
        feat_ok = min_dist >= tau
    else:
        min_dist = None
        feat_ok = True
    filters["feature_distance"] = {
        "min_distance_to_existing": min_dist,
        "threshold": tau,
        "passed": bool(feat_ok),
    }

    is_eligible = (
        filters["numerical"]["passed"]
        and filters["difficulty"]["passed"]
        and filters["learnability"]["passed"]
        and filters["feature_distance"]["passed"]
    )
    return CandidateEvaluation(
        spec=spec,
        raw_feature_vector=raw_feat,
        feature_vector=norm_feat,
        projected_gradients=projected,
        proxy_metrics=proxy_metrics,
        learnability_improvement=learnability,
        filter_results=filters,
        is_eligible=is_eligible,
    )


# ---------------------------------------------------------------------------
# Filter diagnostics / quota fallback
# ---------------------------------------------------------------------------

_FILTER_ORDER = ("numerical", "difficulty", "learnability", "feature_distance")


def build_filter_report(evals: List[CandidateEvaluation]) -> Dict:
    """Per-filter and per-regime pass diagnostics (proposal section 7.5)."""
    report: Dict = {"n_proposed": len(evals)}

    passing = {name: 0 for name in _FILTER_ORDER}
    cumulative = {name: 0 for name in _FILTER_ORDER}
    for ev in evals:
        alive = True
        for name in _FILTER_ORDER:
            ok = bool(ev.filter_results.get(name, {}).get("passed", False))
            if ok:
                passing[name] += 1
            alive = alive and ok
            if alive:
                cumulative[name] += 1
    report["passing_each_filter"] = passing
    report["cumulative_after_filter"] = cumulative
    report["n_eligible"] = sum(1 for ev in evals if ev.is_eligible)

    per_regime: Dict[str, Dict] = {}
    for ev in evals:
        key = str(ev.spec.regime_index)
        d = per_regime.setdefault(key, {"proposed": 0, "eligible": 0})
        d["proposed"] += 1
        d["eligible"] += int(ev.is_eligible)
    for d in per_regime.values():
        d["pass_rate"] = d["eligible"] / max(d["proposed"], 1)
    report["per_regime"] = per_regime

    report["proxy_mase_distribution"] = sorted(
        float(ev.proxy_metrics["mase"])
        for ev in evals
        if ev.proxy_metrics.get("mase") is not None
    )
    report["learnability_distribution"] = sorted(
        float(ev.learnability_improvement)
        for ev in evals
        if ev.learnability_improvement is not None
    )
    return report


def promote_for_quota(
    evals: List[CandidateEvaluation],
    quota_by_regime: Dict[int, int],
) -> List[str]:
    """Deterministic threshold-relaxation fallback for under-quota regimes.

    Promotes the best ineligible candidates (numerically valid first, then by
    learnability, then by candidate id) until each regime meets its quota.
    Returns the promoted candidate ids. Never duplicates candidates and never
    silently skips a grid cell — if a regime still cannot meet quota, raises.
    """
    promoted: List[str] = []
    for regime, quota in sorted(quota_by_regime.items()):
        pool = [ev for ev in evals if ev.spec.regime_index == regime]
        eligible = [ev for ev in pool if ev.is_eligible]
        if len(eligible) >= quota:
            continue
        fallback = sorted(
            (ev for ev in pool if not ev.is_eligible),
            key=lambda ev: (
                not ev.filter_results["numerical"]["passed"],
                -(ev.learnability_improvement
                  if ev.learnability_improvement is not None else -float("inf")),
                ev.spec.candidate_id,
            ),
        )
        need = quota - len(eligible)
        usable = [
            ev for ev in fallback if ev.filter_results["numerical"]["passed"]
        ]
        if len(usable) < need:
            raise RuntimeError(
                f"regime {regime}: cannot meet quota {quota} even after "
                f"relaxation ({len(eligible)} eligible, "
                f"{len(usable)} numerically valid fallbacks)"
            )
        for ev in usable[:need]:
            ev.is_eligible = True
            ev.filter_results["promoted_by_quota_fallback"] = True
            promoted.append(ev.spec.candidate_id)
            print(
                f"[gvendi][WARNING] regime {regime}: promoted "
                f"{ev.spec.candidate_id} via deterministic quota fallback"
            )
    return promoted
