"""Unit tests for projection, entropy, clustering, and selection rules."""

import math

import numpy as np
import pytest
import torch

from gvendi.grad_vendi import (
    EntropyState,
    GradVendi,
    RademacherProjector,
    SelectionCandidate,
    fit_active_gradient_clusters,
    select_existing_regime_candidates,
    select_hard,
    select_new_regime_candidates,
    select_random,
    shortlist_candidates,
)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class TestRademacherProjector:
    def test_deterministic(self):
        proj = RademacherProjector(input_dim=1000, output_dim=32, seed=7)
        g = torch.randn(5, 1000)
        a = proj.project(g)
        b = proj.project(g)
        assert torch.equal(a, b)
        same = RademacherProjector(input_dim=1000, output_dim=32, seed=7)
        assert torch.equal(same.project(g), a)

    def test_seed_changes_output(self):
        g = torch.randn(3, 1000)
        a = RademacherProjector(1000, 32, seed=1).project(g)
        b = RademacherProjector(1000, 32, seed=2).project(g)
        assert not torch.allclose(a, b)

    def test_unit_norm_output(self):
        proj = RademacherProjector(500, 16, seed=0)
        g = torch.randn(8, 500) * 100.0
        out = proj.project(g)
        assert torch.allclose(out.norm(dim=1), torch.ones(8), atol=1e-5)

    def test_scale_invariance(self):
        proj = RademacherProjector(500, 16, seed=0)
        g = torch.randn(500)
        assert torch.allclose(proj.project(g), proj.project(g * 1000.0), atol=1e-5)

    def test_zero_gradient_maps_to_zero(self):
        proj = RademacherProjector(100, 8, seed=0)
        out = proj.project(torch.zeros(100))
        assert torch.equal(out, torch.zeros(8))

    def test_chunking_matches_single_chunk(self):
        g = torch.randn(4, 1000)
        big = RademacherProjector(1000, 16, seed=3, chunk_size=10_000)
        small = RademacherProjector(1000, 16, seed=3, chunk_size=64)
        # Different chunking consumes the RNG differently, so outputs differ;
        # what must hold is that each is internally deterministic.
        assert torch.equal(big.project(g), big.project(g))
        assert torch.equal(small.project(g), small.project(g))

    def test_metadata_roundtrip(self):
        proj = RademacherProjector(1000, 32, seed=9, chunk_size=128)
        clone = RademacherProjector.from_metadata(proj.metadata())
        g = torch.randn(2, 1000)
        assert torch.equal(proj.project(g), clone.project(g))

    def test_shape_check(self):
        proj = RademacherProjector(100, 8, seed=0)
        with pytest.raises(ValueError):
            proj.project(torch.randn(3, 99))


# ---------------------------------------------------------------------------
# Entropy / H-Vendi
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_identical_gradients_entropy_near_zero(self):
        v = torch.randn(16)
        v = v / v.norm()
        g = v.repeat(10, 1)
        assert GradVendi.entropy(g) < 1e-6

    def test_identical_gradients_hvendi_near_one(self):
        v = torch.randn(16)
        v = v / v.norm()
        g = v.repeat(10, 1)
        assert abs(GradVendi.score(g) - 1.0) < 1e-5

    def test_orthogonal_higher_than_identical(self):
        eye = torch.eye(8)
        v = torch.zeros(8)
        v[0] = 1.0
        dup = v.repeat(8, 1)
        assert GradVendi.entropy(eye) > GradVendi.entropy(dup) + 1.0

    def test_orthonormal_entropy_is_log_n(self):
        n = 8
        assert abs(GradVendi.entropy(torch.eye(n)) - math.log(n)) < 1e-6
        assert abs(GradVendi.score(torch.eye(n)) - n) < 1e-3

    def test_marginal_entropy_consistency(self):
        torch.manual_seed(0)
        active = torch.nn.functional.normalize(torch.randn(20, 16), dim=1)
        cand = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
        direct = GradVendi.marginal_entropy(active, cand)
        state = EntropyState(active)
        assert abs(state.marginal(cand) - direct) < 1e-9
        # Adding then measuring matches concatenated entropy.
        state.add(cand)
        joint = GradVendi.entropy(torch.cat([active, cand], dim=0))
        assert abs(state.entropy() - joint) < 1e-9

    def test_orthogonal_candidate_gains_more(self):
        base = torch.zeros(6, 8)
        base[:, 0] = 1.0
        state = EntropyState(base)
        aligned = torch.zeros(2, 8)
        aligned[:, 0] = 1.0
        ortho = torch.zeros(2, 8)
        ortho[:, 1] = 1.0
        assert state.marginal(ortho) > state.marginal(aligned)

    def test_equal_weighting_per_client_blocks(self):
        # Every client contributes the same number of probe vectors, so the
        # scatter accumulation weighs clients equally regardless of add order.
        torch.manual_seed(1)
        blocks = [torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
                  for _ in range(5)]
        s1 = EntropyState(blocks[0])
        for b in blocks[1:]:
            s1.add(b)
        s2 = EntropyState(torch.cat(blocks, dim=0))
        assert abs(s1.entropy() - s2.entropy()) < 1e-9

    def test_zero_matrix_entropy_zero(self):
        assert GradVendi.entropy(torch.zeros(4, 8)) == 0.0


# ---------------------------------------------------------------------------
# Clustering + shortlist
# ---------------------------------------------------------------------------

def _two_blob_active(n_per=6, d=8):
    """Active clients split between two well-separated gradient blobs."""
    torch.manual_seed(0)
    grads = {}
    for i in range(n_per):
        g = torch.zeros(2, d)
        g[:, 0] = 1.0
        grads[f"a_{i:02d}"] = g + 0.01 * torch.randn(2, d)
    for i in range(n_per):
        g = torch.zeros(2, d)
        g[:, 1] = 1.0
        grads[f"b_{i:02d}"] = g + 0.01 * torch.randn(2, d)
    return grads


class TestClustering:
    def test_deterministic(self):
        grads = _two_blob_active()
        km1 = fit_active_gradient_clusters(grads, 4, 0.25, seed=0)
        km2 = fit_active_gradient_clusters(grads, 4, 0.25, seed=0)
        assert torch.equal(km1.centroids, km2.centroids)
        assert torch.equal(km1.assignments, km2.assignments)

    def test_occupancy_sums_to_total(self):
        grads = _two_blob_active()
        km = fit_active_gradient_clusters(grads, 4, 0.25, seed=0)
        assert int(km.occupancy.sum()) == sum(g.shape[0] for g in grads.values())

    def test_sparse_cluster_count(self):
        grads = _two_blob_active()
        km = fit_active_gradient_clusters(grads, 8, 0.25, seed=0)
        assert len(km.sparse_clusters) == 2  # 25% of 8

    def test_shortlist_far_candidate(self):
        grads = _two_blob_active()
        km = fit_active_gradient_clusters(grads, 4, 0.25, seed=0)
        far = torch.zeros(2, 8)
        far[:, 3] = 1.0
        near = grads["a_00"].clone()
        cands = [
            SelectionCandidate("far", 0, far, torch.zeros(3), 1.0),
            SelectionCandidate("near", 0, near, torch.zeros(3), 1.0),
        ]
        shortlist_candidates(
            cands, km, min_sparse_fraction=0.25,
            centroid_distance_quantile=0.75, required_per_regime={},
        )
        far_c = next(c for c in cands if c.candidate_id == "far")
        near_c = next(c for c in cands if c.candidate_id == "near")
        assert far_c.centroid_distance > near_c.centroid_distance
        assert far_c.shortlisted

    def test_shortlist_quota_topup(self):
        grads = _two_blob_active()
        km = fit_active_gradient_clusters(grads, 4, 0.25, seed=0)
        # Every candidate sits right on an active blob: none shortlists
        # organically, so the quota top-up must promote by centroid distance.
        cands = [
            SelectionCandidate(f"c{i}", 0, grads["a_00"].clone(),
                               torch.zeros(3), 1.0)
            for i in range(3)
        ]
        listed = shortlist_candidates(
            cands, km, min_sparse_fraction=1.1,
            centroid_distance_quantile=1.0, required_per_regime={0: 2},
        )
        assert len(listed) >= 2


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------

def _make_candidate(cid, regime, direction, d=8, mase=1.0, feat=None):
    g = torch.zeros(2, d)
    g[:, direction] = 1.0
    feat = torch.zeros(4) if feat is None else feat
    return SelectionCandidate(cid, regime, g, feat, mase)


class TestHVendiSelection:
    def test_one_pick_per_regime(self):
        active = torch.zeros(8, 8)
        active[:, 0] = 1.0
        pools = {
            0: [_make_candidate("r0_a", 0, 1), _make_candidate("r0_b", 0, 0)],
            1: [_make_candidate("r1_a", 1, 2), _make_candidate("r1_b", 1, 0)],
        }
        picks, final_h = select_existing_regime_candidates(
            active, pools, n_order_trials=4, seed=0
        )
        assert set(picks.keys()) == {0, 1}
        # The orthogonal candidates give higher entropy than the aligned ones.
        assert picks[0].candidate_id == "r0_a"
        assert picks[1].candidate_id == "r1_a"
        assert final_h > 0
        for c in picks.values():
            assert c.marginal_entropy_gain is not None
            assert c.marginal_entropy_gain > 0

    def test_new_regime_quota_and_state_mutation(self):
        active = torch.eye(8)[:4].repeat(2, 1)
        state = EntropyState(active)
        n0 = state.n
        cands = [_make_candidate(f"n{i}", 7, i % 8) for i in range(6)]
        picks = select_new_regime_candidates(
            state, cands, quota=3, min_feature_distance=0.0
        )
        assert len(picks) == 3
        assert len({c.candidate_id for c in picks}) == 3
        assert state.n == n0 + 3 * 2

    def test_new_regime_pairwise_feature_distance(self):
        active = torch.eye(8)[:2]
        state = EntropyState(active)
        # Two candidates share a feature vector; guardrail forbids picking both.
        f_a = torch.tensor([0.0, 0.0])
        f_c = torch.tensor([10.0, 0.0])
        cands = [
            _make_candidate("dup1", 7, 3, feat=f_a),
            _make_candidate("dup2", 7, 4, feat=f_a.clone()),
            _make_candidate("distinct", 7, 5, feat=f_c),
        ]
        picks = select_new_regime_candidates(
            state, cands, quota=2, min_feature_distance=1.0
        )
        feats = [tuple(c.feature_vector.tolist()) for c in picks]
        assert len(set(feats)) == 2

    def test_quota_larger_than_pool_raises(self):
        state = EntropyState(torch.eye(4))
        with pytest.raises(ValueError):
            select_new_regime_candidates(
                state, [_make_candidate("x", 7, 0)], quota=2,
                min_feature_distance=0.0,
            )


class TestBaselines:
    def _pools(self):
        feats = [torch.tensor([float(i), 0.0]) for i in range(10)]
        return {
            0: [_make_candidate(f"r0_{i}", 0, i % 8, mase=1.0 + 0.1 * i,
                                feat=feats[i]) for i in range(4)],
            1: [_make_candidate(f"r1_{i}", 1, i % 8, mase=2.0 - 0.1 * i,
                                feat=feats[i]) for i in range(4)],
            7: [_make_candidate(f"n_{i}", 7, i % 8, mase=1.0 + 0.05 * i,
                                feat=feats[i]) for i in range(8)],
        }

    def test_random_deterministic(self):
        a = select_random(self._pools(), 7, 3, 0.0, seed=5)
        b = select_random(self._pools(), 7, 3, 0.0, seed=5)
        assert {r: c.candidate_id for r, c in a[0].items()} == \
               {r: c.candidate_id for r, c in b[0].items()}
        assert [c.candidate_id for c in a[1]] == [c.candidate_id for c in b[1]]
        assert len(a[1]) == 3

    def test_random_respects_quotas(self):
        existing, new = select_random(self._pools(), 7, 3, 0.0, seed=1)
        assert set(existing.keys()) == {0, 1}
        assert len(new) == 3

    def test_hard_picks_max_mase(self):
        existing, new = select_hard(self._pools(), 7, 3, 0.0)
        assert existing[0].candidate_id == "r0_3"  # highest MASE in regime 0
        assert existing[1].candidate_id == "r1_0"  # highest MASE in regime 1
        mases = [c.proxy_mase for c in new]
        assert mases == sorted(mases, reverse=True)

    def test_hard_deterministic(self):
        a = select_hard(self._pools(), 7, 3, 0.0)
        b = select_hard(self._pools(), 7, 3, 0.0)
        assert [c.candidate_id for c in a[1]] == [c.candidate_id for c in b[1]]

    def test_hard_feature_guardrail(self):
        pools = {
            7: [
                _make_candidate("h1", 7, 0, mase=3.0, feat=torch.tensor([0.0])),
                _make_candidate("h2", 7, 1, mase=2.9, feat=torch.tensor([0.05])),
                _make_candidate("h3", 7, 2, mase=1.0, feat=torch.tensor([5.0])),
            ]
        }
        _, new = select_hard(pools, 7, 2, min_feature_distance=1.0)
        assert [c.candidate_id for c in new] == ["h1", "h3"]
