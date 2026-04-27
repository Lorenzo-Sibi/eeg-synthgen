from __future__ import annotations

from itertools import combinations
from math import pi, sqrt

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

_SIGNAL_FREQ_RANGES: dict[str, tuple[float, float]] = {
    "erp": (1.0, 10.0),
    "oscillatory_burst": (4.0, 40.0),
    "ar_correlated": (2.0, 20.0),
    "spike_interictal": (0.5, 3.0),
}


def _pairwise_distances_mm(vertex_coords: np.ndarray, indices: list[int]) -> list[float]:
    if len(indices) < 2:
        return []
    pts = vertex_coords[indices]
    return [float(np.linalg.norm(pts[i] - pts[j])) for i, j in combinations(range(len(pts)), 2)]


def _sample_frequencies(signal_family: str, n: int, rng: np.random.Generator) -> list[float]:
    lo, hi = _SIGNAL_FREQ_RANGES.get(signal_family, (1.0, 40.0))
    return [float(rng.uniform(lo, hi)) for _ in range(n)]


def _grow_patch_bfs(adjacency: sp.spmatrix, seed: int, target_n: int) -> np.ndarray:
    visited: set[int] = {seed}
    frontier: list[int] = [seed]
    while len(visited) < target_n and frontier:
        next_frontier: list[int] = []
        for v in frontier:
            for nb in adjacency.getrow(v).indices:
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
    return np.array(sorted(visited)[:target_n], dtype=np.int32)


def _weighted_adjacency_mm(adjacency: sp.spmatrix, vertex_coords: np.ndarray) -> sp.csr_matrix:
    """Return a CSR graph with edge weights = Euclidean distance (mm) between endpoints.

    Used as an approximation of geodesic distance on the cortical mesh; exact
    on flat triangulations and within a few percent on curved cortex at
    ico5-level vertex spacing.
    """
    adj = adjacency.tocoo(copy=False)
    lengths = np.linalg.norm(
        vertex_coords[adj.row] - vertex_coords[adj.col], axis=1
    ).astype(np.float64)
    return sp.csr_matrix((lengths, (adj.row, adj.col)), shape=adj.shape)


def _grow_patch_geodesic(
    adjacency: sp.spmatrix,
    vertex_coords: np.ndarray,
    seed: int,
    radius_mm: float,
) -> np.ndarray:
    """Return vertex indices within ``radius_mm`` geodesic distance of ``seed``.

    Distance is measured along the cortical mesh with edge weights set to the
    Euclidean length of each edge in mm. The seed itself is always included.
    """
    if radius_mm <= 0.0:
        return np.array([seed], dtype=np.int32)
    w_adj = _weighted_adjacency_mm(adjacency, vertex_coords)
    dist = dijkstra(w_adj, directed=False, indices=seed, limit=float(radius_mm))
    patch = np.where(np.isfinite(dist))[0]
    if seed not in patch:
        patch = np.concatenate([patch, np.array([seed])])
    return np.sort(patch).astype(np.int32)


def _extent_cm2_to_radius_mm(extent_cm2: float) -> float:
    """Convert a requested patch area (cm^2) to a geodesic radius (mm).

    Uses the disk approximation: area = pi * r^2, so r = sqrt(area / pi).
    Converts cm -> mm at the end. Returns 0.0 for non-positive extents.
    """
    if extent_cm2 <= 0.0:
        return 0.0
    return 10.0 * sqrt(extent_cm2 / pi)


def _compute_patches(
    source_space,
    seeds: list[int],
    extents_cm2: list[float],
) -> list[list[int]]:
    """Grow one geodesic patch per (seed, extent_cm2) pair.

    Seed vertex is always included even when the requested extent is tiny.
    """
    adj = source_space.adjacency
    vc = source_space.vertex_coords
    patches: list[list[int]] = []
    for seed, extent in zip(seeds, extents_cm2):
        radius_mm = _extent_cm2_to_radius_mm(float(extent))
        patch = _grow_patch_geodesic(adj, vc, int(seed), radius_mm)
        patches.append([int(v) for v in patch])
    return patches
