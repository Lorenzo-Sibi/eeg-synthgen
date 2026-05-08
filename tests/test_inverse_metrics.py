from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from synthgen.analysis.inverse_metrics import (
    SurfaceGeometry,
    get_patch,
    localization_error,
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


# ---------------------------------------------------------------------------
# localization_error tests (Task 2)
# ---------------------------------------------------------------------------


def _two_seed_setup():
    """5-vertex chain with coords along the x-axis (suitable for 1-hop integer distances)."""
    adj = _make_chain_adjacency(5)
    coords = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]],
        dtype=float,
    )
    return adj, coords


def test_le_perfect_estimate_is_zero():
    adj, coords = _two_seed_setup()
    j_true = np.zeros((5, 10))
    j_true[0, 5] = 1.0
    j_hat = j_true.copy()
    out = localization_error(j_true, j_hat, np.array([0]), coords, adj, patch_order=1)
    assert out["le_mm_mean"] == 0.0
    assert out["per_seed_le_mm"] == [0.0]
    assert out["pred_seed_indices"] == [0]


def test_le_distance_matches_euclidean_within_eval_zone():
    adj, coords = _two_seed_setup()
    j_true = np.zeros((5, 10))
    j_true[0, 5] = 1.0
    j_hat = np.zeros((5, 10))
    j_hat[1, 5] = 1.0  # neighbor of seed 0
    out = localization_error(j_true, j_hat, np.array([0]), coords, adj, patch_order=1)
    assert out["per_seed_le_mm"] == [1.0]
    assert out["pred_seed_indices"] == [1]


def test_le_eval_zone_constraint_ignores_far_argmax():
    """If pred argmax is OUTSIDE the eval zone, it should be ignored."""
    adj, coords = _two_seed_setup()
    j_true = np.zeros((5, 10))
    j_true[0, 5] = 1.0
    j_hat = np.zeros((5, 10))
    j_hat[4, 5] = 100.0  # huge but outside patch_order=1 around seed 0
    j_hat[1, 5] = 1.0    # within patch, smaller
    out = localization_error(j_true, j_hat, np.array([0]), coords, adj, patch_order=1)
    assert out["pred_seed_indices"] == [1]   # picks 1, not 4
    assert out["per_seed_le_mm"] == [1.0]


def test_le_multi_seed_averages():
    adj, coords = _two_seed_setup()
    j_true = np.zeros((5, 10))
    j_true[0, 5] = 1.0
    j_true[4, 5] = 1.0
    j_hat = j_true.copy()  # perfect
    out = localization_error(j_true, j_hat, np.array([0, 4]), coords, adj, patch_order=1)
    assert out["le_mm_mean"] == 0.0
    assert len(out["per_seed_le_mm"]) == 2


def test_le_shape_check_raises():
    adj, coords = _two_seed_setup()
    j_true = np.zeros((5,))
    j_hat = np.zeros((5, 10))
    with pytest.raises(ValueError, match=r"j_true must have shape"):
        localization_error(j_true, j_hat, np.array([0]), coords, adj)


def test_le_scalar_seed_index_does_not_crash():
    adj, coords = _two_seed_setup()
    j_true = np.zeros((5, 10))
    j_true[2, 5] = 1.0
    j_hat = j_true.copy()
    out = localization_error(j_true, j_hat, np.array(2), coords, adj, patch_order=1)
    assert out["le_mm_mean"] == 0.0
    assert out["pred_seed_indices"] == [2]


def test_le_empty_seed_indices_raises():
    adj, coords = _two_seed_setup()
    j_true = np.zeros((5, 10))
    j_hat = np.zeros((5, 10))
    with pytest.raises(ValueError, match="seed_indices is empty"):
        localization_error(j_true, j_hat, np.array([], dtype=int), coords, adj)
