from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from synthgen.sample import Scenario, SourceSpace


def _make_source_space(N: int = 100) -> SourceSpace:
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((N, 3)).astype(np.float32) * 50.0  # mm scale
    row = np.arange(N)
    col = (row + 1) % N  # ring connectivity
    data = np.ones(N, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    adj = adj + adj.T
    parcellation = (np.arange(N) // (N // 4)).astype(np.int32)  # 4 parcels
    hemisphere = (np.arange(N) >= N // 2).astype(np.int32)
    return SourceSpace(
        vertex_coords=coords,
        adjacency=adj,
        parcellation=parcellation,
        hemisphere=hemisphere,
    )


def _make_scenario(**kwargs) -> Scenario:
    defaults = dict(
        scenario_id="test-0",
        seed=42,
        anatomy_id="fsaverage",
        leadfield_id="fsaverage__standard_1005_64__standard",
        montage_id="standard_1005_64",
        reference_scheme="average",
        conductivity_id="standard",
        prior_family="broad_random",
        n_sources=2,
        signal_family="erp",
        split="train",
    )
    defaults.update(kwargs)
    return Scenario(**defaults)


# ── BroadRandomPrior ──────────────────────────────────────────────────────────

def test_broad_random_fills_seed_vertex_indices():
    from synthgen.sources.priors.broad_random import BroadRandomPrior
    rng = np.random.default_rng(0)
    ss = _make_source_space(N=100)
    sc = _make_scenario(n_sources=3)
    prior = BroadRandomPrior()
    sc = prior.sample(sc, ss, rng)
    assert len(sc.seed_vertex_indices) == 3
    assert all(0 <= idx < 100 for idx in sc.seed_vertex_indices)


def test_broad_random_fills_all_fields():
    from synthgen.sources.priors.broad_random import BroadRandomPrior
    rng = np.random.default_rng(0)
    ss = _make_source_space(N=100)
    sc = _make_scenario(n_sources=2)
    prior = BroadRandomPrior()
    sc = prior.sample(sc, ss, rng)
    assert len(sc.patch_extents_cm2) == 2
    assert len(sc.temporal_onsets_s) == 2
    assert len(sc.dominant_frequencies_hz) == 2
    assert isinstance(sc.source_correlation, float)
    assert 0.0 <= sc.source_correlation <= 1.0
    assert all(0.5 <= e <= 10.0 for e in sc.patch_extents_cm2)
    assert all(0.0 <= t <= 0.5 for t in sc.temporal_onsets_s)
    assert len(sc.seed_patch_vertex_indices) == 2
    for seed, patch in zip(sc.seed_vertex_indices, sc.seed_patch_vertex_indices):
        assert seed in patch, "seed must be in its patch"
        assert len(patch) >= 1


def test_broad_random_inter_source_distances_single_source():
    from synthgen.sources.priors.broad_random import BroadRandomPrior
    rng = np.random.default_rng(0)
    ss = _make_source_space(N=100)
    sc = _make_scenario(n_sources=1)
    prior = BroadRandomPrior()
    sc = prior.sample(sc, ss, rng)
    assert sc.inter_source_distances_mm == []


def test_broad_random_inter_source_distances_three_sources():
    from synthgen.sources.priors.broad_random import BroadRandomPrior
    rng = np.random.default_rng(0)
    ss = _make_source_space(N=100)
    sc = _make_scenario(n_sources=3)
    prior = BroadRandomPrior()
    sc = prior.sample(sc, ss, rng)
    # C(3,2) = 3 pairwise distances
    assert len(sc.inter_source_distances_mm) == 3
    assert all(d >= 0.0 for d in sc.inter_source_distances_mm)
    assert any(d > 0.0 for d in sc.inter_source_distances_mm)


def test_broad_random_reproducible_with_same_seed():
    from synthgen.sources.priors.broad_random import BroadRandomPrior
    ss = _make_source_space(N=100)
    prior = BroadRandomPrior()
    sc1 = prior.sample(_make_scenario(n_sources=2), ss, np.random.default_rng(42))
    sc2 = prior.sample(_make_scenario(n_sources=2), ss, np.random.default_rng(42))
    assert sc1.seed_vertex_indices == sc2.seed_vertex_indices
    assert sc1.dominant_frequencies_hz == sc2.dominant_frequencies_hz


def test_grow_patch_bfs_exact_count():
    from synthgen.sources.priors._helpers import _grow_patch_bfs
    N = 20
    row = np.arange(N)
    col = (row + 1) % N
    data = np.ones(N, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    adj = adj + adj.T
    patch = _grow_patch_bfs(adj, seed=0, target_n=7)
    assert len(patch) == 7
    assert 0 in patch


def test_extent_cm2_to_radius_mm_disk_formula():
    from synthgen.sources.priors._helpers import _extent_cm2_to_radius_mm
    # pi cm^2 disk -> 1 cm = 10 mm radius
    assert _extent_cm2_to_radius_mm(np.pi) == pytest.approx(10.0, rel=1e-6)
    assert _extent_cm2_to_radius_mm(0.0) == 0.0
    assert _extent_cm2_to_radius_mm(-1.0) == 0.0


def test_grow_patch_geodesic_includes_seed_and_neighbours():
    from synthgen.sources.priors._helpers import _grow_patch_geodesic
    # 1D chain: 0-1-2-3-4 at 5mm spacing along x
    N = 5
    coords = np.zeros((N, 3), dtype=np.float32)
    coords[:, 0] = np.arange(N) * 5.0
    row = np.arange(N - 1)
    col = row + 1
    data = np.ones(N - 1, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    adj = adj + adj.T
    # radius 12 mm from seed=0 reaches vertices at 0, 5, 10 (not 15)
    patch = _grow_patch_geodesic(adj, coords, seed=0, radius_mm=12.0)
    assert sorted(patch.tolist()) == [0, 1, 2]
    # radius 0 returns just the seed
    patch_zero = _grow_patch_geodesic(adj, coords, seed=2, radius_mm=0.0)
    assert patch_zero.tolist() == [2]


def test_compute_patches_shape_matches_seeds():
    from synthgen.sources.priors._helpers import _compute_patches
    ss = _make_source_space(N=50)
    seeds = [0, 7, 23]
    extents = [1.0, 5.0, 0.0]
    patches = _compute_patches(ss, seeds, extents)
    assert len(patches) == 3
    for seed, patch in zip(seeds, patches):
        assert seed in patch, "seed must always belong to its patch"
        assert all(0 <= v < 50 for v in patch)


def test_sample_frequencies_in_range():
    from synthgen.sources.priors._helpers import _sample_frequencies
    rng = np.random.default_rng(0)
    freqs = _sample_frequencies("oscillatory_burst", n=50, rng=rng)
    assert len(freqs) == 50
    assert all(4.0 <= f <= 40.0 for f in freqs)


# ── LocalContiguousPrior ──────────────────────────────────────────────────────

def test_local_contiguous_fills_all_fields():
    from synthgen.sources.priors.local_contiguous import LocalContiguousPrior
    rng = np.random.default_rng(0)
    ss = _make_source_space(N=100)
    sc = _make_scenario(n_sources=2)
    prior = LocalContiguousPrior()
    sc = prior.sample(sc, ss, rng)
    assert len(sc.seed_vertex_indices) == 2
    assert len(sc.patch_extents_cm2) == 2
    assert len(sc.temporal_onsets_s) == 2
    assert len(sc.dominant_frequencies_hz) == 2
    assert len(sc.inter_source_distances_mm) == 1  # C(2,2)=1 for n_sources=2
    assert 0.0 <= sc.source_correlation <= 0.3


def test_local_contiguous_seed_indices_in_range():
    from synthgen.sources.priors.local_contiguous import LocalContiguousPrior
    rng = np.random.default_rng(1)
    ss = _make_source_space(N=200)
    sc = _make_scenario(n_sources=4)
    prior = LocalContiguousPrior()
    sc = prior.sample(sc, ss, rng)
    assert all(0 <= idx < 200 for idx in sc.seed_vertex_indices)


def test_local_contiguous_patch_extents_within_range():
    from synthgen.sources.priors.local_contiguous import (
        LocalContiguousPrior,
        _EXTENT_CM2_RANGE,
    )
    ss = _make_source_space(N=100)
    prior = LocalContiguousPrior()
    extents = [
        prior.sample(_make_scenario(n_sources=1), ss, np.random.default_rng(i)).patch_extents_cm2[0]
        for i in range(30)
    ]
    lo, hi = _EXTENT_CM2_RANGE
    assert all(lo <= e <= hi for e in extents)


def test_local_contiguous_correlation_low():
    from synthgen.sources.priors.local_contiguous import LocalContiguousPrior
    rng = np.random.default_rng(0)
    ss = _make_source_space(N=100)
    prior = LocalContiguousPrior()
    correlations = []
    for _ in range(30):
        sc = prior.sample(_make_scenario(n_sources=2), ss, rng)
        correlations.append(sc.source_correlation)
    assert max(correlations) <= 0.3


# ── NetworkAwarePrior ─────────────────────────────────────────────────────────

def test_network_aware_fills_all_fields():
    from synthgen.sources.priors.network_aware import NetworkAwarePrior
    rng = np.random.default_rng(0)
    ss = _make_source_space(N=100)
    sc = _make_scenario(n_sources=2)
    prior = NetworkAwarePrior()
    sc = prior.sample(sc, ss, rng)
    assert len(sc.seed_vertex_indices) == 2
    assert len(sc.patch_extents_cm2) == 2
    assert len(sc.dominant_frequencies_hz) == 2
    assert len(sc.inter_source_distances_mm) == 1
    assert len(sc.temporal_onsets_s) == 2
    assert 0.3 <= sc.source_correlation <= 0.8


def test_network_aware_seeds_in_same_parcel():
    from synthgen.sources.priors.network_aware import NetworkAwarePrior
    ss = _make_source_space(N=100)  # 4 parcels of 25 vertices each
    prior = NetworkAwarePrior()
    same_parcel_count = 0
    for seed_val in range(20):
        sc = _make_scenario(n_sources=2, seed=seed_val)
        sc = prior.sample(sc, ss, np.random.default_rng(seed_val))
        parcels = ss.parcellation[sc.seed_vertex_indices]
        if len(set(parcels.tolist())) == 1:
            same_parcel_count += 1
    # Most runs should place seeds in the same parcel
    assert same_parcel_count >= 15


def test_network_aware_correlation_higher_than_local():
    from synthgen.sources.priors.network_aware import NetworkAwarePrior
    from synthgen.sources.priors.local_contiguous import LocalContiguousPrior
    ss = _make_source_space(N=100)
    net_prior = NetworkAwarePrior()
    loc_prior = LocalContiguousPrior()
    net_corrs = [
        net_prior.sample(_make_scenario(n_sources=2), ss, np.random.default_rng(i)).source_correlation
        for i in range(30)
    ]
    loc_corrs = [
        loc_prior.sample(_make_scenario(n_sources=2), ss, np.random.default_rng(i)).source_correlation
        for i in range(30)
    ]
    assert np.mean(net_corrs) > np.mean(loc_corrs)


# ── StateDependentPrior ───────────────────────────────────────────────────────

def test_state_dependent_fills_all_fields():
    from synthgen.sources.priors.state_dependent import StateDependentPrior
    rng = np.random.default_rng(0)
    ss = _make_source_space(N=100)
    sc = _make_scenario(n_sources=2)
    prior = StateDependentPrior()
    sc = prior.sample(sc, ss, rng)
    assert len(sc.seed_vertex_indices) == 2
    assert len(sc.dominant_frequencies_hz) == 2
    assert len(sc.temporal_onsets_s) == 2
    assert len(sc.patch_extents_cm2) == 2
    assert len(sc.inter_source_distances_mm) == 1
    assert isinstance(sc.source_correlation, float)


def test_state_dependent_frequencies_in_valid_range():
    from synthgen.sources.priors.state_dependent import StateDependentPrior
    rng = np.random.default_rng(3)
    ss = _make_source_space(N=100)
    prior = StateDependentPrior()
    all_freqs = []
    for _ in range(50):
        sc = prior.sample(_make_scenario(n_sources=1), ss, rng)
        all_freqs.extend(sc.dominant_frequencies_hz)
    assert all(0.5 <= f <= 30.0 for f in all_freqs)


def test_state_dependent_signal_family_overridden():
    from synthgen.sources.priors.state_dependent import StateDependentPrior, _BRAIN_STATES
    rng = np.random.default_rng(0)
    ss = _make_source_space(N=100)
    prior = StateDependentPrior()
    valid_families = {s["signal_family"] for s in _BRAIN_STATES.values()}
    for _ in range(30):
        sc = _make_scenario(n_sources=1, signal_family="erp")
        sc = prior.sample(sc, ss, rng)
        assert sc.signal_family in valid_families


def test_state_dependent_reproducible():
    from synthgen.sources.priors.state_dependent import StateDependentPrior
    ss = _make_source_space(N=100)
    prior = StateDependentPrior()
    sc1 = prior.sample(_make_scenario(n_sources=2), ss, np.random.default_rng(99))
    sc2 = prior.sample(_make_scenario(n_sources=2), ss, np.random.default_rng(99))
    assert sc1.seed_vertex_indices == sc2.seed_vertex_indices
    assert sc1.dominant_frequencies_hz == sc2.dominant_frequencies_hz
    assert sc1.signal_family == sc2.signal_family
