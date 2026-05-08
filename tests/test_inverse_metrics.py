from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from synthgen.analysis.inverse_metrics import (
    SurfaceGeometry,
    get_patch,
)


def _make_chain_adjacency(n: int):
    """Build CSR adjacency for an open chain 0—1—2—...—(n-1)."""
    rows, cols = [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    data = np.ones(len(rows), dtype=int)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def test_get_patch_order_zero_returns_seed_only():
    adj = _make_chain_adjacency(5)
    patch = get_patch(seed_idx=2, adjacency=adj, order=0)
    assert patch.tolist() == [2]


def test_get_patch_order_one_returns_seed_and_first_neighbors():
    adj = _make_chain_adjacency(5)
    patch = get_patch(seed_idx=2, adjacency=adj, order=1)
    assert patch.tolist() == [1, 2, 3]


def test_get_patch_order_two_grows_two_hops():
    adj = _make_chain_adjacency(5)
    patch = get_patch(seed_idx=2, adjacency=adj, order=2)
    assert patch.tolist() == [0, 1, 2, 3, 4]


def test_surface_geometry_is_frozen_dataclass():
    geom = SurfaceGeometry(
        vertex_coords_mm=np.zeros((3, 3), dtype=np.float32),
        vertex_areas_mm2=np.ones(3, dtype=np.float32),
    )
    with pytest.raises(FrozenInstanceError):
        geom.vertex_coords_mm = np.ones((3, 3))
