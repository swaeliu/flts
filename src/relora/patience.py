"""Per-client patience tracking for the ReLoRA curriculum."""

from __future__ import annotations


class ClientPatienceTracker:
    """Tracks patience and best validation loss independently for each client.

    A client is marked *converged* for the current curriculum round once its
    patience counter reaches `patience`.  The server advances to the next
    curriculum stage only when all active clients have converged, checked via
    `all_converged(n_total)`.

    Call `reset()` at the start of each new curriculum round.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.patience = patience
        self.min_delta = min_delta

        self._best_loss: dict[int, float] = {}
        self._counter: dict[int, int] = {}
        self._converged: set[int] = set()
        self._sampled: set[int] = set()  # all clients seen this round

    # ------------------------------------------------------------------
    # Per-round lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state for a new curriculum round."""
        self._best_loss.clear()
        self._counter.clear()
        self._converged.clear()
        self._sampled.clear()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, client_id: int, val_loss: float) -> tuple[bool, bool]:
        """Record a validation result for *client_id*.

        Returns:
            (improved, converged)
            - improved: True if the client achieved a new best loss this step.
            - converged: True if the client just hit its patience limit
              (monotonically False after the first time it becomes True).
        """
        self._sampled.add(client_id)

        if client_id not in self._best_loss:
            self._best_loss[client_id] = val_loss
            self._counter[client_id] = 0
            return True, False

        if val_loss < self._best_loss[client_id] - self.min_delta:
            self._best_loss[client_id] = val_loss
            self._counter[client_id] = 0
            return True, False

        self._counter[client_id] += 1
        if self._counter[client_id] >= self.patience:
            self._converged.add(client_id)
            return False, True

        return False, False

    # ------------------------------------------------------------------
    # Convergence queries
    # ------------------------------------------------------------------

    def all_converged(self, n_total: int) -> bool:
        """True when every one of the *n_total* active clients has exhausted patience."""
        return len(self._converged) >= n_total

    def convergence_fraction(self, sampled_clients: list[int] | None = None) -> float:
        """Fraction of *sampled_clients* that have converged.

        If *sampled_clients* is None, uses all clients seen so far this round.
        Clients that have not yet been sampled are excluded from the denominator.
        """
        pool = self._sampled if sampled_clients is None else set(sampled_clients)
        if not pool:
            return 0.0
        n_converged = sum(1 for c in pool if c in self._converged)
        return n_converged / len(pool)

    def is_converged(self, client_id: int) -> bool:
        return client_id in self._converged

    @property
    def n_converged(self) -> int:
        return len(self._converged)

    @property
    def n_sampled(self) -> int:
        return len(self._sampled)

    def __repr__(self) -> str:
        return (
            f"ClientPatienceTracker("
            f"patience={self.patience}, "
            f"converged={self.n_converged}/{self.n_sampled})"
        )
