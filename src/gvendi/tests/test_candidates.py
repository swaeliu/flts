"""Unit tests for candidate pool generation and quality-gate plumbing."""

import numpy as np
import pytest
import torch

from gvendi.candidates import (
    CandidateEvaluation,
    CandidateSpec,
    _strategy_counts,
    build_filter_report,
    generate_candidate_pool,
    params_are_valid,
    promote_for_quota,
    sanitize_generator_params,
)
from gvendi.config import GVendiConfig
from gvendi.data import make_grid_params


@pytest.fixture
def cfg():
    c = GVendiConfig()
    c.seed = 0
    c.gvendi_candidates_per_existing_regime = 10
    c.gvendi_candidates_new_regime = 12
    return c


class TestSanitize:
    def test_clips_out_of_range(self):
        params = make_grid_params(GVendiConfig(), 0, 0, 7)
        params["noise_scale"] = 99.0
        params["season_amp_1"] = -3.0
        params["heteroskedasticity"] = 7.0
        out = sanitize_generator_params(params)
        assert out["noise_scale"] == 5.0
        assert out["season_amp_1"] == 0.0
        assert out["heteroskedasticity"] == 2.0

    def test_ar_stability_rescale(self):
        params = make_grid_params(GVendiConfig(), 0, 0, 7)
        params["ar_coef_1"], params["ar_coef_2"], params["ar_coef_3"] = 0.9, 0.9, 0.9
        out = sanitize_generator_params(params)
        total = abs(out["ar_coef_1"]) + abs(out["ar_coef_2"]) + abs(out["ar_coef_3"])
        assert total < 0.95 + 1e-6

    def test_params_are_valid(self):
        params = make_grid_params(GVendiConfig(), 1, 0, 7)
        assert params_are_valid(params)
        params["trend_slope"] = float("nan")
        assert not params_are_valid(params)


class TestStrategyCounts:
    def test_sums_to_n(self, cfg):
        for n in (1, 5, 10, 16, 17):
            counts = _strategy_counts(n, cfg)
            assert sum(counts.values()) == n
            assert all(v >= 0 for v in counts.values())


class TestPoolGeneration:
    def test_counts_and_quotas(self, cfg):
        specs = generate_candidate_pool(cfg, 7, 7, 8, 7)
        by_regime = {}
        for s in specs:
            by_regime.setdefault(s.regime_index, []).append(s)
        for r in range(7):
            assert len(by_regime[r]) == cfg.gvendi_candidates_per_existing_regime
        assert len(by_regime[7]) == cfg.gvendi_candidates_new_regime
        assert len(specs) == 7 * 10 + 12

    def test_unique_ids_and_seeds(self, cfg):
        specs = generate_candidate_pool(cfg, 7, 7, 8, 7)
        ids = [s.candidate_id for s in specs]
        seeds = [s.data_seed for s in specs]
        assert len(set(ids)) == len(ids)
        assert len(set(seeds)) == len(seeds)

    def test_deterministic(self, cfg):
        a = generate_candidate_pool(cfg, 3, 3, 4, 3)
        b = generate_candidate_pool(cfg, 3, 3, 4, 3)
        assert [s.to_dict() for s in a] == [s.to_dict() for s in b]

    def test_all_params_valid(self, cfg):
        specs = generate_candidate_pool(cfg, 3, 3, 4, 3)
        for s in specs:
            assert params_are_valid(s.generator_parameters), s.candidate_id
            assert "family" in s.generator_parameters

    def test_extra_batch_disjoint(self, cfg):
        base = generate_candidate_pool(cfg, 3, 3, 4, 3)
        extra = generate_candidate_pool(cfg, 3, 3, 4, 3, extra_batch=1, regimes=[1])
        base_ids = {s.candidate_id for s in base}
        assert all(s.candidate_id not in base_ids for s in extra)
        assert all(s.regime_index == 1 for s in extra)
        base_seeds = {s.data_seed for s in base}
        assert all(s.data_seed not in base_seeds for s in extra)

    def test_new_regime_family_override(self, cfg):
        cfg.gvendi_new_regime_family = "smooth_seasonal"
        specs = generate_candidate_pool(cfg, 3, 3, 4, 3)
        new = [s for s in specs if s.regime_index == 3]
        assert all(
            s.generator_parameters["family"] == "smooth_seasonal" for s in new
        )

    def test_anchor_weights_change_perturbations(self, cfg):
        a = generate_candidate_pool(cfg, 3, 3, 4, 3)
        b = generate_candidate_pool(
            cfg, 3, 3, 4, 3, anchor_weights={(0, 2): 5.0}
        )
        # Weighted anchoring shifts which variants seed regime 0's candidates.
        assert [s.to_dict() for s in a] != [s.to_dict() for s in b]

    def test_strategy_mixture_present(self, cfg):
        specs = generate_candidate_pool(cfg, 7, 7, 8, 7)
        strategies = {s.generation_strategy for s in specs}
        assert "perturb" in strategies
        assert "interpolate" in strategies
        assert "new_regime" in strategies


# ---------------------------------------------------------------------------
# Filter report + quota fallback
# ---------------------------------------------------------------------------

def _fake_eval(cid, regime, eligible, mase=1.0, learn=0.1, numerical=True):
    spec = CandidateSpec(
        candidate_id=cid,
        regime_index=regime,
        generator_parameters={"family": "trend_seasonal"},
        generation_strategy="perturb",
        data_seed=0,
    )
    filters = {
        "numerical": {"passed": numerical},
        "difficulty": {"passed": eligible, "proxy_mase": mase},
        "learnability": {"passed": eligible, "improvement": learn},
        "feature_distance": {"passed": True},
    }
    return CandidateEvaluation(
        spec=spec,
        raw_feature_vector=torch.zeros(3),
        feature_vector=torch.zeros(3),
        projected_gradients=torch.zeros(2, 4),
        proxy_metrics={"mase": mase},
        learnability_improvement=learn,
        filter_results=filters,
        is_eligible=eligible and numerical,
    )


class TestFilterReport:
    def test_counts(self):
        evals = [
            _fake_eval("a", 0, True),
            _fake_eval("b", 0, False),
            _fake_eval("c", 1, True),
            _fake_eval("d", 1, False, numerical=False),
        ]
        report = build_filter_report(evals)
        assert report["n_proposed"] == 4
        assert report["n_eligible"] == 2
        assert report["passing_each_filter"]["numerical"] == 3
        assert report["cumulative_after_filter"]["numerical"] == 3
        assert report["cumulative_after_filter"]["feature_distance"] == 2
        assert report["per_regime"]["0"]["pass_rate"] == 0.5
        assert len(report["proxy_mase_distribution"]) == 4


class TestQuotaFallback:
    def test_no_promotion_when_quota_met(self):
        evals = [_fake_eval("a", 0, True), _fake_eval("b", 0, False)]
        promoted = promote_for_quota(evals, {0: 1})
        assert promoted == []
        assert not evals[1].is_eligible

    def test_promotes_best_ineligible(self):
        evals = [
            _fake_eval("a", 0, False, learn=0.01),
            _fake_eval("b", 0, False, learn=0.9),
        ]
        promoted = promote_for_quota(evals, {0: 1})
        assert promoted == ["b"]
        assert evals[1].is_eligible
        assert evals[1].filter_results["promoted_by_quota_fallback"]

    def test_never_promotes_numerically_invalid(self):
        evals = [_fake_eval("a", 0, False, numerical=False)]
        with pytest.raises(RuntimeError):
            promote_for_quota(evals, {0: 1})
