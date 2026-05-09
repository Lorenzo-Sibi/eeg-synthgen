"""Pure numpy/scipy metric functions for evaluating EEG inverse methods.

All functions accept source-space row indices (0..V-1) into arrays of
shape (V, T). No MNE, no zarr, no I/O — fully testable in isolation.

Reference: Reynaud et al. (2024), Front. Neurosci. 18:1444935.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceGeometry:
    """Per-vertex geometry of a source space.

    Note: ``frozen=True`` prevents re-binding the attributes, but does
    NOT prevent in-place mutation of the underlying numpy arrays
    (e.g. ``geom.vertex_coords_mm[0, 0] = 1.0`` will silently succeed).
    Callers that need a defensive copy should pass copies in.
    """
    vertex_coords_mm: np.ndarray   # (V, 3), float
    vertex_areas_mm2: np.ndarray   # (V,),    float


def get_patch(seed_idx: int, adjacency, order: int = 2) -> np.ndarray:
    """Order-k local patch around `seed_idx` via BFS over CSR adjacency.

    `adjacency` must be a scipy CSR matrix as returned by
    `mne.spatial_src_adjacency(src).tocsr()`.
    """
    patch = {int(seed_idx)}
    frontier = {int(seed_idx)}
    for _ in range(order):
        new_frontier = set()
        for idx in frontier:
            neighbors = adjacency.indices[adjacency.indptr[idx]:adjacency.indptr[idx + 1]]
            new_frontier.update(int(n) for n in neighbors)
        patch.update(new_frontier)
        frontier = new_frontier
    return np.array(sorted(patch), dtype=int)


def localization_error(
    j_true: np.ndarray,
    j_hat: np.ndarray,
    seed_indices: np.ndarray,
    coords_mm: np.ndarray,
    adjacency,
    patch_order: int = 2,
) -> dict:
    """Localization Error per Reynaud et al. (eq. 7).

    For each seed s:
      t_eval_gt = argmax_t |j_true[s, :]|
      eval_zone = patch order=patch_order around s
      s_hat     = eval_zone[argmax(|j_hat[eval_zone, t_eval_gt]|)]
      LE_s      = ||coords_mm[s] - coords_mm[s_hat]||
    """
    j_true = np.asarray(j_true, dtype=float)
    j_hat = np.asarray(j_hat, dtype=float)
    coords_mm = np.asarray(coords_mm, dtype=float)
    seed_indices = np.asarray(seed_indices, dtype=int)
    seed_indices = np.atleast_1d(seed_indices)
    if seed_indices.size == 0:
        raise ValueError("seed_indices is empty; at least one seed is required")

    if j_true.ndim != 2:
        raise ValueError(f"j_true must have shape (V, T), got {j_true.shape}")
    if j_hat.ndim != 2:
        raise ValueError(f"j_hat must have shape (V, T), got {j_hat.shape}")
    if j_true.shape != j_hat.shape:
        raise ValueError(
            f"j_true and j_hat must have same shape, got {j_true.shape} and {j_hat.shape}"
        )
    if coords_mm.shape[0] != j_true.shape[0]:
        raise ValueError(
            f"coords_mm has {coords_mm.shape[0]} vertices, but j_true has {j_true.shape[0]}"
        )

    le_values: list[float] = []
    pred_seed_indices: list[int] = []
    t_eval_gt_indices: list[int] = []

    for seed_idx in seed_indices:
        seed_idx = int(seed_idx)
        t_eval_gt = int(np.argmax(np.abs(j_true[seed_idx, :])))
        eval_zone = get_patch(seed_idx, adjacency, order=patch_order)
        local_values = np.abs(j_hat[eval_zone, t_eval_gt])
        pred_idx = int(eval_zone[np.argmax(local_values)])
        le_mm = float(np.linalg.norm(coords_mm[seed_idx] - coords_mm[pred_idx]))
        le_values.append(le_mm)
        pred_seed_indices.append(pred_idx)
        t_eval_gt_indices.append(t_eval_gt)

    return {
        "le_mm_mean": float(np.mean(le_values)),
        "per_seed_le_mm": le_values,
        "true_seed_indices": seed_indices.tolist(),
        "pred_seed_indices": pred_seed_indices,
        "t_eval_gt_indices": t_eval_gt_indices,
    }


def time_error(
    j_true: np.ndarray,
    j_hat: np.ndarray,
    seed_indices: np.ndarray,
    adjacency,
    sfreq: float,
    patch_order: int = 2,
) -> dict:
    """Time error per Reynaud et al. (sec. 2.2.4).

    TE_s = |t_eval_gt - t_eval_pred| converted from samples to ms.
    t_eval_pred = argmax_t |j_hat[s_hat, :]| where s_hat is the eval-zone
    argmax used by `localization_error`.
    """
    j_true = np.asarray(j_true, dtype=float)
    j_hat = np.asarray(j_hat, dtype=float)
    seed_indices = np.atleast_1d(np.asarray(seed_indices, dtype=int))
    if seed_indices.size == 0:
        raise ValueError("seed_indices is empty; at least one seed is required")
    if j_true.shape != j_hat.shape or j_true.ndim != 2:
        raise ValueError(
            f"j_true and j_hat must both have shape (V, T); got {j_true.shape}, {j_hat.shape}"
        )
    if sfreq <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")

    te_values: list[float] = []
    t_eval_gt_indices: list[int] = []
    t_eval_pred_indices: list[int] = []
    sample_period_ms = 1000.0 / float(sfreq)

    for seed_idx in seed_indices:
        seed_idx = int(seed_idx)
        t_gt = int(np.argmax(np.abs(j_true[seed_idx, :])))
        eval_zone = get_patch(seed_idx, adjacency, order=patch_order)
        s_hat = int(eval_zone[np.argmax(np.abs(j_hat[eval_zone, t_gt]))])
        t_pred = int(np.argmax(np.abs(j_hat[s_hat, :])))
        te_values.append(abs(t_gt - t_pred) * sample_period_ms)
        t_eval_gt_indices.append(t_gt)
        t_eval_pred_indices.append(t_pred)

    return {
        "te_ms_mean": float(np.mean(te_values)),
        "per_seed_te_ms": te_values,
        "t_eval_gt_indices": t_eval_gt_indices,
        "t_eval_pred_indices": t_eval_pred_indices,
    }
