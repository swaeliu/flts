"""Atomic persistence, manifest validation, and resume detection.

Every artifact write goes through a temp-file + rename so a mid-run crash
(previously observed as disk-quota failures) never leaves a truncated JSON or
checkpoint behind. Only projected gradients, candidate metadata, cluster
state, and manifests are stored — never full unprojected gradients.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List

import numpy as np
import torch

SCHEMA_VERSION = 1


class _NumpyJSONEncoder(json.JSONEncoder):
    """Serialize the numpy scalars that generator params / stats can contain."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

def atomic_write_json(path: str, obj) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, cls=_NumpyJSONEncoder)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_torch_save(path: str, obj) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage-directory layout
# ---------------------------------------------------------------------------

class GVendiPaths:
    """File layout inside <out_dir>/gvendi_stage_<k>/.

    The candidate pool, probe gradients, cluster state, and filter report are
    shared across every selection method; the manifest and report are
    per-method so Random / Hard / H-Vendi runs can coexist in one stage dir.
    """

    def __init__(self, out_dir: str, stage_index: int) -> None:
        self.stage_dir = os.path.join(out_dir, f"gvendi_stage_{stage_index}")

    def ensure(self) -> "GVendiPaths":
        os.makedirs(self.stage_dir, exist_ok=True)
        return self

    @property
    def candidate_pool(self) -> str:
        return os.path.join(self.stage_dir, "candidate_pool.json")

    @property
    def active_gradients(self) -> str:
        return os.path.join(self.stage_dir, "active_gradients.pt")

    @property
    def candidate_gradients(self) -> str:
        return os.path.join(self.stage_dir, "candidate_gradients.pt")

    @property
    def kmeans_state(self) -> str:
        return os.path.join(self.stage_dir, "kmeans_state.pt")

    @property
    def candidate_evals(self) -> str:
        return os.path.join(self.stage_dir, "candidate_evals.pt")

    @property
    def filter_report(self) -> str:
        return os.path.join(self.stage_dir, "filter_report.json")

    def selection_manifest(self, method: str) -> str:
        return os.path.join(self.stage_dir, f"selection_manifest_{method}.json")

    def selection_report(self, method: str) -> str:
        return os.path.join(self.stage_dir, f"selection_report_{method}.json")

    def train_dir(self, method: str) -> str:
        return os.path.join(self.stage_dir, f"train_{method}")


# ---------------------------------------------------------------------------
# Candidate pool persistence
# ---------------------------------------------------------------------------

def save_candidate_pool(path: str, specs: List, cfg) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_pool_seed": cfg.gvendi_candidate_seed,
        "base_seed": cfg.seed,
        "candidates": [s.to_dict() for s in specs],
    }
    atomic_write_json(path, payload)


def load_candidate_pool(path: str) -> List[Dict]:
    payload = load_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"candidate pool {path} has schema_version "
            f"{payload.get('schema_version')}, expected {SCHEMA_VERSION}"
        )
    return payload["candidates"]


# ---------------------------------------------------------------------------
# Selection manifest
# ---------------------------------------------------------------------------

_REQUIRED_CLIENT_FIELDS = (
    "client_key", "candidate_id", "regime_index", "variant_index",
    "generator_parameters", "data_seed",
)


def build_selection_manifest(
    cfg,
    method: str,
    source_stage: str,
    target_stage: str,
    source_checkpoint: str,
    selected_rows: List[Dict],
) -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_stage": source_stage,
        "target_stage": target_stage,
        "selection_method": method,
        "source_checkpoint": source_checkpoint,
        "candidate_pool_seed": cfg.gvendi_candidate_seed,
        "projection_seed": cfg.gvendi_projection_seed,
        "probe_seed": cfg.gvendi_probe_seed,
        "selection_seed": cfg.gvendi_selection_seed,
        "projection_dim": cfg.gvendi_projection_dim,
        "probe_batches": cfg.gvendi_probe_batches,
        "probe_batch_size": cfg.gvendi_probe_batch_size,
        "num_clusters": cfg.gvendi_num_clusters,
        "sparse_cluster_fraction": cfg.gvendi_sparse_cluster_fraction,
        "quality_thresholds": {
            "mase_low_quantile": cfg.gvendi_mase_low_quantile,
            "mase_high_quantile": cfg.gvendi_mase_high_quantile,
            "min_learnability": cfg.gvendi_min_learnability,
            "min_feature_distance": cfg.gvendi_min_existing_feature_distance,
            "min_selected_feature_distance": cfg.gvendi_min_selected_feature_distance,
        },
        "selected_clients": selected_rows,
    }


def validate_selection_manifest(
    manifest: Dict,
    method: str,
    prev_n_regimes: int,
    prev_n_variants: int,
    new_n_regimes: int,
    new_n_variants: int,
) -> None:
    """Raise if a manifest is unusable for the requested run configuration."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version {manifest.get('schema_version')} "
            f"!= {SCHEMA_VERSION}"
        )
    if manifest.get("selection_method") != method:
        raise ValueError(
            f"manifest selection_method {manifest.get('selection_method')!r} "
            f"!= requested {method!r}"
        )
    rows = manifest.get("selected_clients", [])
    # Every cell of the new grid that is not a cell of the previous grid must
    # come from the manifest (covers both the +1 regime / +1 variant expansion
    # and the variant-only expansion).
    expected_cells = {
        (r, v)
        for r in range(new_n_regimes)
        for v in range(new_n_variants)
        if not (r < prev_n_regimes and v < prev_n_variants)
    }
    expected = len(expected_cells)
    if len(rows) != expected:
        raise ValueError(
            f"manifest has {len(rows)} selected clients, expected {expected}"
        )
    cells = set()
    for row in rows:
        for f in _REQUIRED_CLIENT_FIELDS:
            if f not in row:
                raise ValueError(f"selected client missing field {f!r}: {row}")
        cells.add((int(row["regime_index"]), int(row["variant_index"])))
    if cells != expected_cells:
        raise ValueError(
            f"manifest grid cells {sorted(cells)} != expected "
            f"{sorted(expected_cells)}"
        )


def try_load_selection_manifest(
    path: str,
    method: str,
    prev_n_regimes: int,
    prev_n_variants: int,
    new_n_regimes: int,
    new_n_variants: int,
) -> Dict | None:
    """Return a validated manifest, or None when absent/invalid (re-select)."""
    if not os.path.exists(path):
        return None
    try:
        manifest = load_json(path)
        validate_selection_manifest(
            manifest, method,
            prev_n_regimes, prev_n_variants, new_n_regimes, new_n_variants,
        )
        return manifest
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[gvendi][WARNING] ignoring invalid selection manifest {path}: {exc}")
        return None


def selected_client_row(
    evaluation,
    selection_candidate,
    regime_index: int,
    variant_index: int,
) -> Dict:
    """Serialize one selected candidate into a manifest row."""
    spec = evaluation.spec
    return {
        "client_key": f"regime_{regime_index:02d}_variant_{variant_index:02d}",
        "candidate_id": spec.candidate_id,
        "regime_index": regime_index,
        "variant_index": variant_index,
        "generator_parameters": dict(spec.generator_parameters),
        "generation_strategy": spec.generation_strategy,
        "source_variants": list(spec.source_variants),
        "data_seed": spec.data_seed,
        "client_features": [float(v) for v in evaluation.feature_vector.tolist()],
        "proxy_mase": (
            float(evaluation.proxy_metrics["mase"])
            if evaluation.proxy_metrics.get("mase") is not None else None
        ),
        "learnability_improvement": (
            float(evaluation.learnability_improvement)
            if evaluation.learnability_improvement is not None else None
        ),
        "sparse_fraction": float(selection_candidate.sparse_fraction),
        "centroid_distance": float(selection_candidate.centroid_distance),
        "marginal_entropy_gain": (
            float(selection_candidate.marginal_entropy_gain)
            if selection_candidate.marginal_entropy_gain is not None else None
        ),
    }
