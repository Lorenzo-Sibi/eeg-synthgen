#!/usr/bin/env python3
"""Build structural-connectivity bank aligned with the anatomy parcellation.

Two code paths are supported:

1. **Surrogate projection** (default): start from a TVB bundled connectome
   (66 / 68 / 76 / 96 / 192 regions), then project to the target parcellation
   by nearest-centroid mapping. Use for DK / Destrieux / HCP-MMP1 / Schaefer
   and any other scheme that lacks a matched connectome.

2. **Native** (DeepSIF-994 only): load DeepSIF's `connectivity_998.zip`
   (Hagmann 2008 style, TVB-format ZIP) and project onto the 994-region DeepSIF
   parcellation. Activated automatically when `--scheme deepsif_994` is used
   and the DeepSIF atlas directory contains `connectivity_998.zip`.

Usage:
    # DK parcellation, TVB-192 surrogate base
    python scripts/prepare_connectivity.py --config config/default.yaml \\
        --anatomy fsaverage --scheme desikan_killiany --tvb-base 192

    # DeepSIF-994 parcellation, native 998 connectome
    python scripts/prepare_connectivity.py --config config/default.yaml \\
        --anatomy fsaverage --scheme deepsif_994 \\
        --deepsif-dir banks/atlases/deepsif
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from synthgen.banks.anatomy import AnatomyBank
from synthgen.config import GenerationConfig


_TVB_BASE_CHOICES = (66, 68, 76, 96, 192)


def _region_centers(vertex_coords: np.ndarray, parcellation: np.ndarray, n_regions: int) -> np.ndarray:
    centers = np.zeros((n_regions, 3), dtype=np.float64)
    counts = np.zeros(n_regions, dtype=np.int64)
    for r in range(n_regions):
        m = parcellation == r
        if m.any():
            centers[r] = vertex_coords[m].mean(axis=0)
            counts[r] = m.sum()
    if (counts == 0).any():
        empty = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Empty parcels in parcellation: {empty}")
    return centers.astype(np.float32)


def _load_tvb_bundled(n_regions: int):
    if n_regions not in _TVB_BASE_CHOICES:
        raise ValueError(f"--tvb-base must be one of {_TVB_BASE_CHOICES}, got {n_regions}")
    import tvb_data.connectivity as tc
    from tvb.datatypes.connectivity import Connectivity as TVBConnectivity
    zip_path = os.path.join(os.path.dirname(tc.__file__), f"connectivity_{n_regions}.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"TVB connectome not found: {zip_path}")
    return TVBConnectivity.from_file(zip_path)


def _load_deepsif_998(deepsif_dir: Path):
    # TVB's Connectivity.from_file resolves relative paths against tvb_data's
    # own root, not the cwd, so always pass an absolute path.
    zip_path = (deepsif_dir / "connectivity_998.zip").resolve()
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Run: python scripts/fetch_atlases.py --deepsif"
        )
    from tvb.datatypes.connectivity import Connectivity as TVBConnectivity
    return TVBConnectivity.from_file(str(zip_path))


def _compute_target_to_base(base_centers: np.ndarray, target_centers: np.ndarray) -> np.ndarray:
    """Assign each target centroid to its nearest base centroid (Euclidean).

    Both centroid sets are translated to zero-mean before matching, which makes
    the projection robust to origin shifts between coordinate frames (TVB
    bundled connectomes ship in a Talairach-/MNI-like frame, while anatomy
    banks store cortex coordinates in MNE's head frame after ``mri_head_t``;
    the z-origin alone differs by ~90 mm). Without centering, the systematic
    offset dominates the Euclidean distance and causes many target parcels to
    collapse onto a small subset of base parcels.

    Returns an int64 array of shape (R_target,) whose entries are indices into
    ``base_centers``. Many-to-one is allowed: several target parcels can map to
    the same base parcel; this is normal whenever the target scheme is finer
    than the base scheme.
    """
    base_centers = np.asarray(base_centers, dtype=np.float32)
    target_centers = np.asarray(target_centers, dtype=np.float32)
    base_centered = base_centers - base_centers.mean(axis=0)
    target_centered = target_centers - target_centers.mean(axis=0)
    R = target_centered.shape[0]
    target_to_base = np.zeros(R, dtype=np.int64)
    for i, c in enumerate(target_centered):
        d = np.linalg.norm(base_centered - c, axis=1)
        target_to_base[i] = int(np.argmin(d))
    return target_to_base


def _project_by_nearest_centroid(base_conn, target_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map target parcels to the nearest base-connectome centroid, then index
    the base weights/tract_lengths matrices. Symmetrize, zero the diagonal, and
    normalize ``W`` so ``max(row_sum) == 1``.

    Normalization matters: TVB's bundled 76 / 96 / 192 connectomes ship raw
    weights in ``[0, 3]`` with unnormalized row sums (up to ~130 for TVB-192),
    which is enough to drive Jansen-Rit into numerical overflow regardless of
    ``global_coupling``. Projecting them to a denser target parcellation
    (DK-69 at ~52% density) makes the problem worse. Rescaling to
    ``max(row_sum) == 1`` puts the bank in the same regime as TVB-68, where
    canonical Jansen-Rit couplings (~0.01-0.02) are stable.
    """
    base_centers = np.asarray(base_conn.centres, dtype=np.float32)
    base_w = np.asarray(base_conn.weights, dtype=np.float32)
    base_tl = np.asarray(base_conn.tract_lengths, dtype=np.float32)

    target_to_base = _compute_target_to_base(base_centers, target_centers)

    W = base_w[np.ix_(target_to_base, target_to_base)].astype(np.float32)
    L = base_tl[np.ix_(target_to_base, target_to_base)].astype(np.float32)
    W = 0.5 * (W + W.T)
    L = 0.5 * (L + L.T)
    np.fill_diagonal(W, 0.0)
    np.fill_diagonal(L, 0.0)

    row_sum_max = float(W.sum(axis=1).max())
    if row_sum_max > 0.0:
        W = (W / row_sum_max).astype(np.float32)
    return W, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--scheme", type=str, default="deepsif_994")
    ap.add_argument("--anatomy", type=str, default="fsaverage")
    ap.add_argument("--tvb-base", type=int, default=192, choices=_TVB_BASE_CHOICES,
                    help="TVB bundled connectome to use as surrogate base (default: 192).")
    ap.add_argument("--deepsif-dir", type=Path, default=Path("banks/atlases/deepsif"),
                    help="Directory containing DeepSIF connectivity_998.zip.")
    args = ap.parse_args()

    cfg = GenerationConfig.from_yaml(args.config)
    anat = AnatomyBank(cfg.anatomy_bank).load(args.anatomy, scheme=args.scheme)
    n_regions = int(anat.parcellation.max()) + 1
    assert len(anat.region_labels) == n_regions, "region_labels length must match parcel count"

    centers = _region_centers(anat.vertex_coords, anat.parcellation, n_regions)

    use_native_deepsif = (args.scheme == "deepsif_994" and (args.deepsif_dir / "connectivity_998.zip").exists())
    if use_native_deepsif:
        base_conn = _load_deepsif_998(args.deepsif_dir)
        base_tag = "deepsif_998"
    else:
        print("Warning! Using surrogate projection from TVB bundled connectome. For a more accurate DeepSIF-994 connectome, run with --scheme deepsif_994 and place connectivity_998.zip in the DeepSIF atlas directory.")
        base_conn = _load_tvb_bundled(args.tvb_base)
        base_tag = f"tvb_{args.tvb_base}"

    W, L = _project_by_nearest_centroid(base_conn, centers)

    out_dir = Path(cfg.connectivity_bank.bank_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.scheme}.npz"
    np.savez_compressed(
        out_path,
        weights=W,
        tract_lengths=L,
        region_centers=centers,
        region_labels=np.array(anat.region_labels, dtype=str),
        scheme=args.scheme,
    )
    print(
        f"Wrote {out_path}  shape={W.shape}  base={base_tag}  "
        f"mean(W)={W.mean():.3e}  mean(L)={L.mean():.2f}mm"
    )


if __name__ == "__main__":
    main()
