"""Unit tests for the new-client sampling policy and split metrics."""

import numpy as np

from gvendi.train_8x8 import (
    new_client_weight,
    sample_clients_weighted,
    split_metrics,
)


class TestNewClientWeight:
    def test_schedule_shape(self):
        # w(t) = 1 + (alpha - 1) * max(0, 1 - t / T)
        assert new_client_weight(0, 2.0, 20) == 2.0
        assert new_client_weight(10, 2.0, 20) == 1.5
        assert new_client_weight(20, 2.0, 20) == 1.0
        assert new_client_weight(100, 2.0, 20) == 1.0

    def test_no_boost(self):
        assert new_client_weight(0, 1.0, 20) == 1.0
        assert new_client_weight(0, 2.0, 0) == 1.0


class TestWeightedSampling:
    def test_cap_enforced(self):
        rng = np.random.default_rng(0)
        new_ids = set(range(32, 64))
        for _ in range(50):
            picked = sample_clients_weighted(rng, 64, new_ids, 8, 100.0, 0.5)
            assert len(picked) == 8
            assert len(set(picked)) == 8
            n_new = sum(1 for c in picked if c in new_ids)
            assert n_new <= 4

    def test_boost_increases_new_frequency(self):
        new_ids = set(range(49, 64))
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(1)
        boosted = sum(
            sum(1 for c in sample_clients_weighted(rng_a, 64, new_ids, 8, 3.0, 0.5)
                if c in new_ids)
            for _ in range(200)
        )
        uniform = sum(
            sum(1 for c in sample_clients_weighted(rng_b, 64, new_ids, 8, 1.0, 0.5)
                if c in new_ids)
            for _ in range(200)
        )
        assert boosted > uniform

    def test_deterministic_given_rng(self):
        new_ids = {5, 6, 7}
        a = sample_clients_weighted(np.random.default_rng(3), 9, new_ids, 4, 2.0, 0.5)
        b = sample_clients_weighted(np.random.default_rng(3), 9, new_ids, 4, 2.0, 0.5)
        assert a == b

    def test_k_larger_than_population(self):
        picked = sample_clients_weighted(
            np.random.default_rng(0), 4, {3}, 8, 2.0, 0.5
        )
        assert sorted(picked) == [0, 1, 2, 3]


class TestSplitMetrics:
    def test_split_and_worst_regime(self):
        per_client = [
            {"client_id": 0, "regime_id": 0, "mae": 1.0, "mase": 1.0},
            {"client_id": 1, "regime_id": 0, "mae": 2.0, "mase": 2.0},
            {"client_id": 2, "regime_id": 1, "mae": 3.0, "mase": 5.0},
        ]
        out = split_metrics(per_client, new_client_ids={2})
        assert out["old_client_mean_mae"] == 1.5
        assert out["new_client_mean_mae"] == 3.0
        assert out["old_client_mean_mase"] == 1.5
        assert out["new_client_mean_mase"] == 5.0
        assert out["worst_regime_mase"] == 5.0

    def test_handles_missing_mase(self):
        per_client = [
            {"client_id": 0, "regime_id": 0, "mae": 1.0, "mase": None},
        ]
        out = split_metrics(per_client, new_client_ids=set())
        assert out["old_client_mean_mase"] is None
        assert out["worst_regime_mase"] is None
