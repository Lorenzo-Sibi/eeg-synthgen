"""Pure numpy/scipy metric functions for evaluating EEG inverse methods.

All functions accept source-space row indices (0..V-1) into arrays of
shape (V, T). No MNE, no zarr, no I/O — fully testable in isolation.

Reference: Reynaud et al. (2024), Front. Neurosci. 18:1444935.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


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


def normalized_mse(
    j_true: np.ndarray,
    j_hat: np.ndarray,
    seed_indices: np.ndarray,
    adjacency,
    patch_order: int = 2,
) -> dict:
    """nMSE per Reynaud et al. (eq. 8) at t_eval_gt for each seed.

        nmse_s = mean_v ( j_true[v, t]/|j_true[:, t]|.max()
                         - j_hat[v, t]/|j_hat[:, t]|.max() )^2

    Mean over seeds. Zero-protected divisions: if the per-time-slice max
    of either signal is zero, that seed contributes NaN.
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

    nmse_values: list[float] = []
    for seed_idx in seed_indices:
        seed_idx = int(seed_idx)
        t_gt = int(np.argmax(np.abs(j_true[seed_idx, :])))
        true_slice = j_true[:, t_gt]
        hat_slice = j_hat[:, t_gt]
        true_max = np.abs(true_slice).max()
        hat_max = np.abs(hat_slice).max()
        if true_max == 0 or hat_max == 0:
            nmse_values.append(float("nan"))
            continue
        diff = (true_slice / true_max) - (hat_slice / hat_max)
        nmse_values.append(float(np.mean(diff ** 2)))
    return {
        "nmse_mean": float(np.nanmean(nmse_values)),
        "per_seed_nmse": nmse_values,
    }


def psnr_temporal(j_true: np.ndarray, j_hat: np.ndarray) -> float:
    """PSNR over the entire (V, T) tensor per Reynaud et al. (eq. 9).

    Both signals are normalized by their max-abs (so values lie in
    [-1, 1]); data_range is fixed at 2.0. Equivalent to:

        PSNR = 10 * log10(MAX^2 / MSE)

    with MAX = 2.0 and MSE computed over all (V, T) entries of the
    normalized arrays.
    """
    j_true = np.asarray(j_true, dtype=float)
    j_hat = np.asarray(j_hat, dtype=float)
    if j_true.shape != j_hat.shape or j_true.ndim != 2:
        raise ValueError(
            f"j_true and j_hat must both have shape (V, T); got {j_true.shape}, {j_hat.shape}"
        )

    true_max = np.abs(j_true).max()
    hat_max = np.abs(j_hat).max()
    if true_max == 0 or hat_max == 0:
        return float("nan")

    norm_true = j_true / true_max
    norm_hat = j_hat / hat_max
    mse = float(np.mean((norm_true - norm_hat) ** 2))
    if mse == 0:
        return float("inf")
    data_range = 2.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def auc_at_peak(
    j_hat: np.ndarray,
    support: np.ndarray,
    seed_indices: np.ndarray,
    j_true: np.ndarray,
    adjacency,
    patch_order: int = 2,
) -> dict:
    """ROC-AUC at the time of GT peak activity (per Reynaud et al.).

    For each seed s:
        t_eval_gt = argmax_t |j_true[s, :]|
        score     = |j_hat[:, t_eval_gt]|     (continuous, shape (V,))
        label     = support                    (binary GT, shape (V,))
        auc_s     = roc_auc_score(label, score)
    Mean over seeds. If support is all-True or all-False the AUC is
    undefined and NaN is returned.

    `adjacency` and `patch_order` are accepted for signature uniformity
    with the other metrics; `auc_at_peak` does NOT use a local eval zone
    (it scores over the whole source space).
    """
    j_hat = np.asarray(j_hat, dtype=float)
    j_true = np.asarray(j_true, dtype=float)
    support = np.asarray(support, dtype=bool)
    seed_indices = np.atleast_1d(np.asarray(seed_indices, dtype=int))
    if seed_indices.size == 0:
        raise ValueError("seed_indices is empty; at least one seed is required")
    if j_hat.shape != j_true.shape or j_hat.ndim != 2:
        raise ValueError(
            f"j_hat and j_true must both have shape (V, T); got {j_hat.shape}, {j_true.shape}"
        )
    if support.shape != (j_hat.shape[0],):
        raise ValueError(
            f"support must have shape (V,)={(j_hat.shape[0],)}, got {support.shape}"
        )

    if not support.any() or support.all():
        return {
            "auc_mean": float("nan"),
            "per_seed_auc": [float("nan")] * len(seed_indices),
        }

    auc_values: list[float] = []
    for seed_idx in seed_indices:
        seed_idx = int(seed_idx)
        t_gt = int(np.argmax(np.abs(j_true[seed_idx, :])))
        score = np.abs(j_hat[:, t_gt])
        auc_values.append(float(roc_auc_score(support, score)))
    return {
        "auc_mean": float(np.mean(auc_values)),
        "per_seed_auc": auc_values,
    }


def compute_all_metrics(
    j_true: np.ndarray,
    j_hat: np.ndarray,
    seed_indices: np.ndarray,
    support: np.ndarray,
    coords_mm: np.ndarray,
    adjacency,
    sfreq: float,
    patch_order: int = 2,
) -> dict:
    """Single-call entry point: returns a flat dict suitable for direct
    insertion as a row in a long-format records DataFrame.
    """
    le = localization_error(j_true, j_hat, seed_indices, coords_mm, adjacency, patch_order)
    te = time_error(j_true, j_hat, seed_indices, adjacency, sfreq, patch_order)
    nm = normalized_mse(j_true, j_hat, seed_indices, adjacency, patch_order)
    ps = psnr_temporal(j_true, j_hat)
    au = auc_at_peak(j_hat, support, seed_indices, j_true, adjacency, patch_order)
    return {
        "le_mm": le["le_mm_mean"],
        "te_ms": te["te_ms_mean"],
        "nmse": nm["nmse_mean"],
        "psnr_db": ps,
        "auc": au["auc_mean"],
        "per_seed_le_mm": le["per_seed_le_mm"],
        "per_seed_te_ms": te["per_seed_te_ms"],
        "per_seed_nmse": nm["per_seed_nmse"],
        "per_seed_auc": au["per_seed_auc"],
        "true_seed_indices": le["true_seed_indices"],
        "pred_seed_indices": le["pred_seed_indices"],
        "t_eval_gt_indices": te["t_eval_gt_indices"],
        "t_eval_pred_indices": te["t_eval_pred_indices"],
    }
