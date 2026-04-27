"""Unit tests for scripts/prepare_connectivity.py helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).parent.parent / "scripts" / "prepare_connectivity.py"
    spec = importlib.util.spec_from_file_location("prepare_connectivity", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_connectivity"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _fake_base(n=10, seed=0):
    """Base connectome with unnormalized weights (TVB-192 style: max~3, uneven rows)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.0, 3.0, size=(n, n)).astype(np.float32)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    L = rng.uniform(20.0, 120.0, size=(n, n)).astype(np.float32)
    L = 0.5 * (L + L.T)
    np.fill_diagonal(L, 0.0)
    centres = rng.standard_normal((n, 3)).astype(np.float32) * 50.0
    return SimpleNamespace(weights=W, tract_lengths=L, centres=centres)


def test_projection_normalizes_row_sum_max_to_one(mod):
    base = _fake_base(n=20)
    target_centers = base.centres[:12]  # project onto 12 parcels
    W, L = mod._project_by_nearest_centroid(base, target_centers)

    assert W.shape == (12, 12)
    assert L.shape == (12, 12)
    # Diagonal must be zero and matrix symmetric
    np.testing.assert_array_equal(np.diag(W), np.zeros(12, dtype=np.float32))
    np.testing.assert_array_equal(np.diag(L), np.zeros(12, dtype=np.float32))
    np.testing.assert_allclose(W, W.T, atol=1e-6)
    # The load-bearing invariant: max row sum is exactly 1 (the normalization)
    assert W.sum(axis=1).max() == pytest.approx(1.0, rel=1e-6)
    # Weights must be non-negative and bounded
    assert W.min() >= 0.0
    assert W.max() <= 1.0


def test_projection_handles_zero_weights(mod):
    """All-zero base weights must not raise (row_sum_max == 0 branch)."""
    base = _fake_base(n=8)
    base.weights = np.zeros_like(base.weights)
    W, L = mod._project_by_nearest_centroid(base, base.centres[:5])
    assert np.all(W == 0.0)
    assert W.sum() == 0.0


def test_compute_target_to_base_picks_nearest(mod):
    """Each target must map to the unambiguously closest base centroid."""
    base_centers = np.array(
        [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
    )
    target_centers = np.array(
        [
            [1.0, 0.0, 0.0],   # closest to base 0
            [48.0, 0.0, 0.0],  # closest to base 1
            [97.0, 0.0, 0.0],  # closest to base 2
            [49.0, 0.0, 0.0],  # closest to base 1 (many-to-one with target 1)
        ],
        dtype=np.float32,
    )
    mapping = mod._compute_target_to_base(base_centers, target_centers)
    assert mapping.tolist() == [0, 1, 2, 1]
    assert mapping.dtype == np.int64


def test_weight_inheritance_one_to_one(mod):
    """When target == base centers (1-to-1), W_target equals base W up to the
    final normalization (row-sum-max scaling) and the diagonal-zero / symmetry
    enforcement that's already on the base."""
    base = _fake_base(n=6)
    W, L = mod._project_by_nearest_centroid(base, base.centres)
    # Recover the un-normalized matrix and compare against base
    scale = base.weights.sum(axis=1).max()  # base is already symmetric, diag 0
    W_unnorm = W * scale
    np.testing.assert_allclose(W_unnorm, base.weights, atol=1e-5)
    np.testing.assert_allclose(L, base.tract_lengths, atol=1e-5)


def test_many_to_one_mapping_shares_rows(mod):
    """Target parcels that map to the same base parcel must inherit *identical*
    rows and columns from the base (up to the global normalization scalar)."""
    # Three base regions; two target parcels both fall closest to base 0.
    base = _fake_base(n=3, seed=7)
    target_centers = np.array(
        [
            [0.1, 0.0, 0.0],   # near base 0
            [0.2, 0.0, 0.0],   # also near base 0
            base.centres[1],   # exactly base 1
            base.centres[2],   # exactly base 2
        ],
        dtype=np.float32,
    )
    # Force the geometry so first two targets map to base 0
    base.centres = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]], dtype=np.float32
    )
    target_centers[2] = base.centres[1]
    target_centers[3] = base.centres[2]

    mapping = mod._compute_target_to_base(base.centres, target_centers)
    assert mapping.tolist() == [0, 0, 1, 2]

    W, _ = mod._project_by_nearest_centroid(base, target_centers)
    # Rows 0 and 1 of W must be identical (both target parcels saw the same
    # base-row contribution before symmetrization, which cannot break this
    # because the partner mappings are identical too).
    np.testing.assert_allclose(W[0, :], W[1, :], atol=1e-6)
    np.testing.assert_allclose(W[:, 0], W[:, 1], atol=1e-6)


def test_compute_target_to_base_invariant_to_origin_shift(mod):
    """The mapping must NOT depend on the absolute origin of either centroid
    set, since base (TVB) and target (anatomy) live in different coordinate
    frames whose origins can differ by ~90 mm. Centering happens internally."""
    base_centers = np.array(
        [[-30.0, 0.0, 0.0], [0.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=np.float32
    )
    target_centers = np.array(
        [[-25.0, 0.0, 0.0], [5.0, 0.0, 0.0], [27.0, 0.0, 0.0]], dtype=np.float32
    )
    expected = mod._compute_target_to_base(base_centers, target_centers)

    # Shift target by an arbitrary translation; mapping must be unchanged
    shift = np.array([100.0, -50.0, 90.0], dtype=np.float32)
    shifted = target_centers + shift
    np.testing.assert_array_equal(
        mod._compute_target_to_base(base_centers, shifted),
        expected,
    )

    # Shift base too; still unchanged
    np.testing.assert_array_equal(
        mod._compute_target_to_base(base_centers + shift, shifted),
        expected,
    )


def test_target_order_permutes_rows_and_cols(mod):
    """Permuting the target ordering must permute rows/cols of the resulting
    matrices the same way — semantics depend only on the set of target
    centroids, not their indexing."""
    base = _fake_base(n=10, seed=11)
    target = base.centres.copy()
    perm = np.array([3, 1, 4, 0, 2, 7, 6, 5, 9, 8])
    W_a, L_a = mod._project_by_nearest_centroid(base, target)
    W_b, L_b = mod._project_by_nearest_centroid(base, target[perm])
    np.testing.assert_allclose(W_b, W_a[np.ix_(perm, perm)], atol=1e-6)
    np.testing.assert_allclose(L_b, L_a[np.ix_(perm, perm)], atol=1e-6)
