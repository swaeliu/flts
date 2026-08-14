"""Gradient projection, entropy (H-Vendi), clustering, and candidate selection.

Pure tensor math — no model or data dependencies — so every function here is
unit-testable and shared by all three selection methods (random / hard /
h_vendi differ only in which ranking function the orchestrator calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Rademacher projection
# ---------------------------------------------------------------------------

class RademacherProjector:
    """Deterministic chunked Rademacher projection R^D -> R^d.

    The full projection matrix (D x d) is never materialised: sign blocks are
    regenerated chunk-by-chunk from a fixed torch.Generator seed, so the
    projector is exactly reproducible from (seed, input_dim, output_dim,
    chunk_size) metadata and identical across runs, methods, and clients.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int,
        chunk_size: int = 65536,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)

    def metadata(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "seed": self.seed,
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_metadata(cls, meta: dict) -> "RademacherProjector":
        return cls(
            input_dim=meta["input_dim"],
            output_dim=meta["output_dim"],
            seed=meta["seed"],
            chunk_size=meta.get("chunk_size", 65536),
        )

    def project(self, gradient: torch.Tensor) -> torch.Tensor:
        """Project [D] or [B, D] gradients to unit-norm [d] / [B, d] vectors.

        Inputs are L2-normalised before projection and the output is
        re-normalised, so only gradient *direction* information survives.
        All-zero gradients map to all-zero outputs (flagged upstream).
        """
        squeeze = gradient.ndim == 1
        g = gradient.reshape(1, -1) if squeeze else gradient
        if g.shape[1] != self.input_dim:
            raise ValueError(
                f"expected input dim {self.input_dim}, got {g.shape[1]}"
            )
        g = g.detach().to(torch.float32).cpu()

        norms = g.norm(dim=1, keepdim=True)
        g = g / (norms + _EPS)

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        out = torch.zeros(g.shape[0], self.output_dim, dtype=torch.float32)
        for start in range(0, self.input_dim, self.chunk_size):
            stop = min(start + self.chunk_size, self.input_dim)
            block = (
                torch.randint(
                    0, 2, (stop - start, self.output_dim), generator=gen
                ).to(torch.float32)
                * 2.0
                - 1.0
            )
            out += g[:, start:stop] @ block

        out = out / (out.norm(dim=1, keepdim=True) + _EPS)
        # Re-zero rows that came from zero gradients (their projection is
        # numerically ~0 already; make it exact).
        out[norms.squeeze(1) < _EPS] = 0.0
        return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Entropy / H-Vendi
# ---------------------------------------------------------------------------

def _entropy_from_scatter(scatter: torch.Tensor, n: int, eps: float = 1e-10) -> float:
    """Eigenvalue entropy of C = scatter / n, with clamped, normalised spectrum."""
    if n <= 0:
        return 0.0
    cov = scatter.to(torch.float64) / float(n)
    lam = torch.linalg.eigvalsh(cov)
    lam = lam.clamp_min(0.0)
    total = lam.sum()
    if total <= eps:
        return 0.0
    lam = lam / (total + eps)
    return float(-(lam * torch.log(lam + eps)).sum().item())


class GradVendi:
    """Entropy and H-Vendi over normalised projected gradient sets [N, d]."""

    @staticmethod
    def entropy(gradients: torch.Tensor) -> float:
        g = gradients.to(torch.float64)
        return _entropy_from_scatter(g.T @ g, g.shape[0])

    @staticmethod
    def score(gradients: torch.Tensor) -> float:
        """H-Vendi diagnostic: exp(entropy). Reported, never used for ranking."""
        return float(np.exp(GradVendi.entropy(gradients)))

    @staticmethod
    def marginal_entropy(
        active_gradients: torch.Tensor,
        candidate_gradients: torch.Tensor,
    ) -> float:
        base = GradVendi.entropy(active_gradients)
        joint = GradVendi.entropy(
            torch.cat([active_gradients, candidate_gradients], dim=0)
        )
        return joint - base


class EntropyState:
    """Incremental entropy over a growing gradient set via its scatter matrix.

    Adding candidate set G_c updates S <- S + G_c^T G_c and N <- N + B, which
    is exact for H(G u G_c) and avoids re-stacking the full matrix on every
    greedy evaluation.
    """

    def __init__(self, gradients: torch.Tensor) -> None:
        g = gradients.to(torch.float64)
        self.scatter = g.T @ g
        self.n = int(g.shape[0])

    def entropy(self) -> float:
        return _entropy_from_scatter(self.scatter, self.n)

    def entropy_with(self, candidate_gradients: torch.Tensor) -> float:
        g = candidate_gradients.to(torch.float64)
        return _entropy_from_scatter(self.scatter + g.T @ g, self.n + g.shape[0])

    def marginal(self, candidate_gradients: torch.Tensor) -> float:
        return self.entropy_with(candidate_gradients) - self.entropy()

    def add(self, candidate_gradients: torch.Tensor) -> None:
        g = candidate_gradients.to(torch.float64)
        self.scatter = self.scatter + g.T @ g
        self.n += int(g.shape[0])

    def clone(self) -> "EntropyState":
        out = EntropyState.__new__(EntropyState)
        out.scatter = self.scatter.clone()
        out.n = self.n
        return out


# ---------------------------------------------------------------------------
# K-means over active projected gradients
# ---------------------------------------------------------------------------

@dataclass
class KMeansState:
    centroids: torch.Tensor          # [K, d]
    assignments: torch.Tensor        # [N] cluster id per active gradient
    occupancy: torch.Tensor          # [K] active-gradient counts
    sparse_clusters: tuple           # cluster ids in the bottom occupancy tier
    active_client_keys: tuple        # client key per active *client*
    active_centroid_distance: torch.Tensor  # [C] mean nearest-centroid dist per client
    active_sparse_fraction: dict = field(default_factory=dict)  # key -> fraction

    def to_payload(self) -> dict:
        return {
            "centroids": self.centroids,
            "assignments": self.assignments,
            "occupancy": self.occupancy,
            "sparse_clusters": list(self.sparse_clusters),
            "active_client_keys": list(self.active_client_keys),
            "active_centroid_distance": self.active_centroid_distance,
            "active_sparse_fraction": dict(self.active_sparse_fraction),
        }


def _kmeans_plus_plus_init(
    x: torch.Tensor, k: int, rng: np.random.Generator
) -> torch.Tensor:
    n = x.shape[0]
    first = int(rng.integers(0, n))
    centroids = [x[first]]
    for _ in range(1, k):
        d2 = torch.stack(
            [((x - c) ** 2).sum(dim=1) for c in centroids], dim=1
        ).min(dim=1).values
        total = float(d2.sum().item())
        if total <= _EPS:
            idx = int(rng.integers(0, n))
        else:
            probs = (d2 / total).cpu().numpy().astype(np.float64)
            probs = probs / probs.sum()
            idx = int(rng.choice(n, p=probs))
        centroids.append(x[idx])
    return torch.stack(centroids, dim=0)


def _kmeans(
    x: torch.Tensor, k: int, seed: int, max_iters: int = 100, tol: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic k-means. Returns (centroids [K, d], assignments [N])."""
    x = x.to(torch.float32)
    n = x.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    centroids = _kmeans_plus_plus_init(x, k, rng).clone()

    assignments = torch.zeros(n, dtype=torch.long)
    for _ in range(max_iters):
        dists = torch.cdist(x, centroids)
        assignments = dists.argmin(dim=1)
        new_centroids = centroids.clone()
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centroids[j] = x[mask].mean(dim=0)
            else:
                # Deterministic empty-cluster repair: move to the point
                # farthest from its assigned centroid.
                far = dists.gather(1, assignments.unsqueeze(1)).squeeze(1).argmax()
                new_centroids[j] = x[far]
        shift = (new_centroids - centroids).norm(dim=1).max().item()
        centroids = new_centroids
        if shift < tol:
            break
    dists = torch.cdist(x, centroids)
    assignments = dists.argmin(dim=1)
    return centroids, assignments


def fit_active_gradient_clusters(
    active_gradients: dict[str, torch.Tensor],
    num_clusters: int,
    sparse_cluster_fraction: float,
    seed: int,
    max_iters: int = 100,
) -> KMeansState:
    """Fit k-means to the pooled active probe gradients.

    Args:
        active_gradients: mapping client_key -> [B, d] projected gradients.
    """
    keys = tuple(sorted(active_gradients.keys()))
    stacked = torch.cat([active_gradients[k] for k in keys], dim=0)
    centroids, assignments = _kmeans(stacked, num_clusters, seed, max_iters)

    k = centroids.shape[0]
    occupancy = torch.bincount(assignments, minlength=k)

    n_sparse = max(int(round(sparse_cluster_fraction * k)), 1)
    order = sorted(range(k), key=lambda j: (int(occupancy[j]), j))
    sparse = tuple(order[:n_sparse])

    # Per-client stats over its own probe block.
    per_client_dist = []
    sparse_fraction: dict[str, float] = {}
    offset = 0
    sparse_set = set(sparse)
    for key in keys:
        b = active_gradients[key].shape[0]
        block_assign = assignments[offset:offset + b]
        block = stacked[offset:offset + b]
        dmin = torch.cdist(block, centroids).min(dim=1).values
        per_client_dist.append(float(dmin.mean().item()))
        sparse_fraction[key] = float(
            sum(int(a) in sparse_set for a in block_assign) / b
        )
        offset += b

    return KMeansState(
        centroids=centroids,
        assignments=assignments,
        occupancy=occupancy,
        sparse_clusters=sparse,
        active_client_keys=keys,
        active_centroid_distance=torch.tensor(per_client_dist, dtype=torch.float32),
        active_sparse_fraction=sparse_fraction,
    )


def candidate_cluster_stats(
    candidate_gradients: torch.Tensor, state: KMeansState
) -> tuple[float, float]:
    """Return (sparse_fraction, mean nearest-centroid distance) for [B, d]."""
    dists = torch.cdist(candidate_gradients.to(torch.float32), state.centroids)
    nearest = dists.argmin(dim=1)
    sparse_set = set(state.sparse_clusters)
    sparse_fraction = float(
        sum(int(a) in sparse_set for a in nearest) / max(len(nearest), 1)
    )
    centroid_distance = float(dists.min(dim=1).values.mean().item())
    return sparse_fraction, centroid_distance


# ---------------------------------------------------------------------------
# Selection candidates + shortlisting
# ---------------------------------------------------------------------------

@dataclass
class SelectionCandidate:
    candidate_id: str
    regime_index: int
    gradients: torch.Tensor      # [B, d] projected probe gradients
    feature_vector: torch.Tensor  # normalised client features
    proxy_mase: float
    sparse_fraction: float = 0.0
    centroid_distance: float = 0.0
    shortlisted: bool = False
    marginal_entropy_gain: float | None = None


def shortlist_candidates(
    candidates: list[SelectionCandidate],
    state: KMeansState,
    min_sparse_fraction: float,
    centroid_distance_quantile: float,
    required_per_regime: dict[int, int],
) -> list[SelectionCandidate]:
    """Annotate candidates with cluster stats and mark the shortlist.

    A candidate survives when its sparse-cluster fraction >= threshold OR its
    mean nearest-centroid distance exceeds the active-client quantile. Regimes
    whose shortlist is smaller than their quota are topped up with the
    remaining eligible candidates of largest centroid distance.
    """
    if len(state.active_centroid_distance) > 0:
        dist_threshold = float(
            torch.quantile(
                state.active_centroid_distance.to(torch.float64),
                centroid_distance_quantile,
            ).item()
        )
    else:
        dist_threshold = float("inf")

    for cand in candidates:
        sf, cd = candidate_cluster_stats(cand.gradients, state)
        cand.sparse_fraction = sf
        cand.centroid_distance = cd
        cand.shortlisted = (sf >= min_sparse_fraction) or (cd >= dist_threshold)

    for regime, quota in required_per_regime.items():
        pool = [c for c in candidates if c.regime_index == regime]
        listed = [c for c in pool if c.shortlisted]
        if len(listed) < quota:
            extras = sorted(
                (c for c in pool if not c.shortlisted),
                key=lambda c: (-c.centroid_distance, c.candidate_id),
            )
            for c in extras[: quota - len(listed)]:
                c.shortlisted = True

    return [c for c in candidates if c.shortlisted]


# ---------------------------------------------------------------------------
# Feature-distance guardrail
# ---------------------------------------------------------------------------

def _violates_pairwise_distance(
    cand: SelectionCandidate,
    chosen: list[SelectionCandidate],
    min_distance: float,
) -> bool:
    if min_distance <= 0.0:
        return False
    for other in chosen:
        d = float((cand.feature_vector - other.feature_vector).norm().item())
        if d < min_distance:
            return True
    return False


# ---------------------------------------------------------------------------
# H-Vendi quota-constrained greedy selection
# ---------------------------------------------------------------------------

def select_existing_regime_candidates(
    active_gradients: torch.Tensor,
    candidates_by_regime: dict[int, list[SelectionCandidate]],
    n_order_trials: int,
    seed: int,
) -> tuple[dict[int, SelectionCandidate], float]:
    """Greedy per-regime argmax of marginal entropy, over randomized regime orders.

    Returns (regime -> chosen candidate, final entropy of the best trial).
    The winning trial's per-selection marginal gains are stored on the chosen
    candidates' ``marginal_entropy_gain``.
    """
    regimes = sorted(candidates_by_regime.keys())
    for r in regimes:
        if not candidates_by_regime[r]:
            raise ValueError(f"regime {r} has no candidates to select from")

    rng = np.random.default_rng(seed)
    orders = [list(rng.permutation(regimes)) for _ in range(max(n_order_trials, 1))]

    best_entropy = -float("inf")
    best_picks: dict[int, SelectionCandidate] | None = None
    best_gains: dict[int, float] | None = None

    base_state = EntropyState(active_gradients)
    for order in orders:
        state = base_state.clone()
        picks: dict[int, SelectionCandidate] = {}
        gains: dict[int, float] = {}
        for regime in order:
            current = state.entropy()
            scored = [
                (state.entropy_with(c.gradients) - current, c)
                for c in candidates_by_regime[regime]
            ]
            scored.sort(key=lambda t: (-t[0], t[1].candidate_id))
            gain, chosen = scored[0]
            picks[regime] = chosen
            gains[regime] = gain
            state.add(chosen.gradients)
        final = state.entropy()
        if final > best_entropy:
            best_entropy = final
            best_picks = picks
            best_gains = gains

    assert best_picks is not None and best_gains is not None
    for regime, cand in best_picks.items():
        cand.marginal_entropy_gain = best_gains[regime]
    return best_picks, best_entropy


def select_new_regime_candidates(
    entropy_state: EntropyState,
    candidates: list[SelectionCandidate],
    quota: int,
    min_feature_distance: float,
) -> list[SelectionCandidate]:
    """Sequential greedy marginal-entropy picks with a pairwise feature guardrail.

    Mutates *entropy_state* by adding each chosen candidate's gradients. When
    every remaining candidate violates the guardrail, the constraint is
    dropped for that pick (deterministic, logged fallback) rather than
    silently under-filling the quota.
    """
    if len(candidates) < quota:
        raise ValueError(
            f"new regime has {len(candidates)} candidates but quota is {quota}"
        )
    chosen: list[SelectionCandidate] = []
    remaining = list(candidates)
    for _ in range(quota):
        current = entropy_state.entropy()
        admissible = [
            c for c in remaining
            if not _violates_pairwise_distance(c, chosen, min_feature_distance)
        ]
        if not admissible:
            print(
                "[gvendi] feature-distance guardrail excluded every remaining "
                "candidate; relaxing constraint for this pick"
            )
            admissible = remaining
        scored = [
            (entropy_state.entropy_with(c.gradients) - current, c)
            for c in admissible
        ]
        scored.sort(key=lambda t: (-t[0], t[1].candidate_id))
        gain, pick = scored[0]
        pick.marginal_entropy_gain = gain
        entropy_state.add(pick.gradients)
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


# ---------------------------------------------------------------------------
# Random and Hard baselines (same quotas, same pools, different ranking only)
# ---------------------------------------------------------------------------

def select_random(
    candidates_by_regime: dict[int, list[SelectionCandidate]],
    new_regime_index: int,
    new_regime_quota: int,
    min_feature_distance: float,
    seed: int,
) -> tuple[dict[int, SelectionCandidate], list[SelectionCandidate]]:
    """Random eligible candidates under the same quotas (fixed baseline seed)."""
    rng = np.random.default_rng(seed)

    existing: dict[int, SelectionCandidate] = {}
    for regime in sorted(candidates_by_regime.keys()):
        if regime == new_regime_index:
            continue
        pool = sorted(candidates_by_regime[regime], key=lambda c: c.candidate_id)
        existing[regime] = pool[int(rng.integers(0, len(pool)))]

    new_pool = sorted(
        candidates_by_regime.get(new_regime_index, []),
        key=lambda c: c.candidate_id,
    )
    chosen: list[SelectionCandidate] = []
    order = list(rng.permutation(len(new_pool)))
    for idx in order:
        cand = new_pool[idx]
        if _violates_pairwise_distance(cand, chosen, min_feature_distance):
            continue
        chosen.append(cand)
        if len(chosen) == new_regime_quota:
            break
    # Deterministic fallback: fill remaining slots ignoring the guardrail,
    # in the same shuffled order.
    if len(chosen) < new_regime_quota:
        print(
            "[gvendi] random baseline: guardrail left quota unfilled; "
            "relaxing constraint for remaining picks"
        )
        for idx in order:
            cand = new_pool[idx]
            if cand in chosen:
                continue
            chosen.append(cand)
            if len(chosen) == new_regime_quota:
                break
    return existing, chosen


def select_hard(
    candidates_by_regime: dict[int, list[SelectionCandidate]],
    new_regime_index: int,
    new_regime_quota: int,
    min_feature_distance: float,
) -> tuple[dict[int, SelectionCandidate], list[SelectionCandidate]]:
    """Highest proxy MASE under the same quotas.

    The quality gate's MASE upper bound has already been applied, so this
    selects difficult but non-pathological candidates.
    """
    existing: dict[int, SelectionCandidate] = {}
    for regime in sorted(candidates_by_regime.keys()):
        if regime == new_regime_index:
            continue
        pool = sorted(
            candidates_by_regime[regime],
            key=lambda c: (-c.proxy_mase, c.candidate_id),
        )
        existing[regime] = pool[0]

    new_pool = sorted(
        candidates_by_regime.get(new_regime_index, []),
        key=lambda c: (-c.proxy_mase, c.candidate_id),
    )
    chosen: list[SelectionCandidate] = []
    for cand in new_pool:
        if _violates_pairwise_distance(cand, chosen, min_feature_distance):
            continue
        chosen.append(cand)
        if len(chosen) == new_regime_quota:
            break
    if len(chosen) < new_regime_quota:
        print(
            "[gvendi] hard baseline: guardrail left quota unfilled; "
            "relaxing constraint for remaining picks"
        )
        for cand in new_pool:
            if cand in chosen:
                continue
            chosen.append(cand)
            if len(chosen) == new_regime_quota:
                break
    return existing, chosen
