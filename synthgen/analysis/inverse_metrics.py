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
