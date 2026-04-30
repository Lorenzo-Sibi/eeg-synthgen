from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from synthgen.config import GenerationConfig
from synthgen.sample import Scenario, SourceSpace


def _make_config(tmp_path) -> GenerationConfig:
    import yaml
    cfg = {
        "anatomy_bank": {"bank_dir": str(tmp_path / "anatomy")},
        "leadfield_bank": {"bank_dir": str(tmp_path / "leadfield")},
        "montages": {"montages": []},
        "writer": {"output_dir": str(tmp_path / "out")},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    return GenerationConfig.from_yaml(p)


def _make_source_space(N: int = 50) -> SourceSpace:
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((N, 3)).astype(np.float32) * 50.0
    adj = sp.eye(N, format="csr", dtype=np.float32)
    return SourceSpace(
        vertex_coords=coords,
        adjacency=adj,
        parcellation=np.zeros(N, dtype=np.int32),
        hemisphere=np.zeros(N, dtype=np.int32),
    )


def _make_scenario(n_sources: int = 2, signal_family: str = "erp") -> Scenario:
    sc = Scenario(
        scenario_id="test-0",
        seed=42,
        anatomy_id="fsaverage",
        leadfield_id="fsaverage__standard_1005_64__standard",
        montage_id="standard_1005_64",
        reference_scheme="average",
        conductivity_id="standard",
        prior_family="broad_random",
        n_sources=n_sources,
        signal_family=signal_family,
        difficulty="easy",
        split="train",
    )
    sc.seed_vertex_indices = list(range(n_sources))
    sc.temporal_onsets_s = [0.0] * n_sources
    sc.dominant_frequencies_hz = [10.0] * n_sources
    return sc


def test_sereega_backend_output_shapes(tmp_path):
    from synthgen.sources.sereega_backend import SEREEGABackend
    config = _make_config(tmp_path)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=50)
    sc = _make_scenario(n_sources=2)
    rng = np.random.default_rng(0)
    src, bg = backend.generate(sc, ss, rng)
    T = config.temporal.n_samples_per_window
    assert src.shape == (50, T)
    assert bg.shape == (50, T)


def test_sereega_backend_source_nonzero_at_seeds(tmp_path):
    from synthgen.sources.sereega_backend import SEREEGABackend
    config = _make_config(tmp_path)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=50)
    sc = _make_scenario(n_sources=2)
    rng = np.random.default_rng(0)
    src, _ = backend.generate(sc, ss, rng)
    for seed_idx in sc.seed_vertex_indices:
        assert np.any(src[seed_idx] != 0.0), f"seed vertex {seed_idx} should be nonzero"


def test_sereega_backend_background_all_vertices_nonzero(tmp_path):
    from synthgen.sources.sereega_backend import SEREEGABackend
    config = _make_config(tmp_path)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=50)
    sc = _make_scenario(n_sources=1)
    rng = np.random.default_rng(0)
    _, bg = backend.generate(sc, ss, rng)
    assert np.all(np.std(bg, axis=1) > 0.0)


@pytest.mark.parametrize("signal_family", ["erp", "oscillatory_burst", "ar_correlated", "spike_interictal"])
def test_sereega_backend_signal_families(tmp_path, signal_family):
    from synthgen.sources.sereega_backend import SEREEGABackend
    config = _make_config(tmp_path)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=20)
    sc = _make_scenario(n_sources=1, signal_family=signal_family)
    rng = np.random.default_rng(0)
    src, bg = backend.generate(sc, ss, rng)
    T = config.temporal.n_samples_per_window
    assert src.shape == (20, T)
    assert np.any(src[0] != 0.0)


def test_sereega_backend_reproducible(tmp_path):
    from synthgen.sources.sereega_backend import SEREEGABackend
    config = _make_config(tmp_path)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=30)
    sc = _make_scenario(n_sources=2)
    src1, bg1 = backend.generate(sc, ss, np.random.default_rng(77))
    src2, bg2 = backend.generate(sc, ss, np.random.default_rng(77))
    np.testing.assert_array_equal(src1, src2)
    np.testing.assert_array_equal(bg1, bg2)


def test_sereega_backend_output_dtype(tmp_path):
    from synthgen.sources.sereega_backend import SEREEGABackend
    config = _make_config(tmp_path)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=20)
    sc = _make_scenario(n_sources=1)
    src, bg = backend.generate(sc, ss, np.random.default_rng(0))
    assert src.dtype == np.float32
    assert bg.dtype == np.float32


def test_sereega_backend_broadcasts_signal_across_patch(tmp_path):
    """Every vertex in seed_patch_vertex_indices[i] must carry the source signal."""
    from synthgen.sources.sereega_backend import SEREEGABackend
    config = _make_config(tmp_path)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=50)
    sc = _make_scenario(n_sources=2)
    # Patch 0: vertices {0, 10, 11}.  Patch 1: vertices {1, 20}.
    sc.seed_patch_vertex_indices = [[0, 10, 11], [1, 20]]
    src, _ = backend.generate(sc, ss, np.random.default_rng(0))
    # Every patch vertex must be nonzero and equal to its seed
    for patch in sc.seed_patch_vertex_indices:
        seed = patch[0]
        for v in patch:
            np.testing.assert_array_equal(src[v], src[seed])
            assert np.any(src[v] != 0.0)
    # Non-patch vertices must be zero
    active = {v for patch in sc.seed_patch_vertex_indices for v in patch}
    for v in range(50):
        if v not in active:
            assert np.all(src[v] == 0.0)
