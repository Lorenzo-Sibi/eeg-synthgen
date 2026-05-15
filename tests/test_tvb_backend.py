import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("tvb")

from synthgen.banks.connectivity import Connectivity
from synthgen.config import (
    AnatomyBankConfig,
    ConnectivityBankConfig,
    GenerationConfig,
    LeadfieldBankConfig,
    MontageConfig,
    TVBBackendConfig,
    TemporalConfig,
    WriterConfig,
)
from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.tvb_backend import TVBSourceGenerator


def _tiny_conn(R=4):
    rng = np.random.default_rng(0)
    W = rng.uniform(0.0, 0.5, (R, R)).astype(np.float32)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return Connectivity(
        weights=W,
        tract_lengths=np.ones((R, R), dtype=np.float32) * 50.0,
        region_centers=np.random.default_rng(1).standard_normal((R, 3)).astype(np.float32),
        region_labels=[f"r{i}" for i in range(R)],
        scheme="test",
    )


def _tiny_source_space(N=40, R=4):
    rng = np.random.default_rng(2)
    parc = np.repeat(np.arange(R, dtype=np.int32), N // R)
    return SourceSpace(
        vertex_coords=rng.standard_normal((N, 3)).astype(np.float32),
        adjacency=sp.eye(N, format="csr", dtype=np.float32),
        parcellation=parc,
        hemisphere=np.zeros(N, dtype=np.int32),
        parcellation_scheme="test",
        region_labels=[f"r{i}" for i in range(R)],
    )


def _tiny_config(tmp_path, window_s=0.2, reservoir_duration_s=2.0, reservoir_size=8):
    return GenerationConfig(
        anatomy_bank=AnatomyBankConfig(bank_dir=tmp_path / "a"),
        leadfield_bank=LeadfieldBankConfig(bank_dir=tmp_path / "l"),
        montages=MontageConfig(),
        connectivity_bank=ConnectivityBankConfig(bank_dir=tmp_path / "c"),
        temporal=TemporalConfig(sfreq=500.0, window_s=window_s),
        writer=WriterConfig(output_dir=tmp_path / "out"),
        backend="tvb",
        tvb=TVBBackendConfig(
            warmup_s=0.2,
            reservoir_duration_s=reservoir_duration_s,
            reservoir_size=reservoir_size,
            integrator_dt_ms=1.0,
        ),
    )


def _scenario(seeds):
    return Scenario(
        scenario_id="s", seed=0, anatomy_id="a", leadfield_id="l",
        montage_id="m", reference_scheme="average", conductivity_id="standard",
        prior_family="broad_random", n_sources=len(seeds),
        signal_family="erp", split="train",
        seed_vertex_indices=list(seeds),
        temporal_onsets_s=[0.05] * len(seeds),
        dominant_frequencies_hz=[10.0] * len(seeds),
    )


def test_generate_shapes_and_dtypes(tmp_path):
    cfg = _tiny_config(tmp_path)
    conn = _tiny_conn()
    backend = TVBSourceGenerator(cfg, conn)
    ss = _tiny_source_space()
    sc = _scenario(seeds=[0, 10])
    s, bg = backend.generate(sc, ss, np.random.default_rng(0))
    T = cfg.temporal.n_samples_per_window
    assert s.shape == (40, T) and bg.shape == (40, T)
    assert s.dtype == np.float32 and bg.dtype == np.float32
    assert np.isfinite(s).all() and np.isfinite(bg).all()


def test_source_activity_zero_outside_seed_parcels(tmp_path):
    cfg = _tiny_config(tmp_path)
    conn = _tiny_conn()
    backend = TVBSourceGenerator(cfg, conn)
    ss = _tiny_source_space()
    sc = _scenario(seeds=[0, 20])
    s, _ = backend.generate(sc, ss, np.random.default_rng(0))
    parc = ss.parcellation
    non_seed = np.isin(parc, [1, 3])
    assert np.all(s[non_seed] == 0.0)


def test_background_activity_dense_nonzero(tmp_path):
    cfg = _tiny_config(tmp_path)
    conn = _tiny_conn()
    backend = TVBSourceGenerator(cfg, conn)
    ss = _tiny_source_space()
    sc = _scenario(seeds=[0])
    _, bg = backend.generate(sc, ss, np.random.default_rng(0))
    assert np.all(bg.std(axis=1) > 0.0)


def test_determinism_same_rng_same_output(tmp_path):
    cfg = _tiny_config(tmp_path)
    conn = _tiny_conn()
    backend = TVBSourceGenerator(cfg, conn)
    ss = _tiny_source_space()
    sc = _scenario(seeds=[0])
    s1, bg1 = backend.generate(sc, ss, np.random.default_rng(123))
    s2, bg2 = backend.generate(sc, ss, np.random.default_rng(123))
    assert np.allclose(s1, s2)
    assert np.allclose(bg1, bg2)


def test_parcellation_scheme_mismatch_raises(tmp_path):
    cfg = _tiny_config(tmp_path)
    conn = _tiny_conn()
    backend = TVBSourceGenerator(cfg, conn)
    ss = _tiny_source_space()
    ss.parcellation_scheme = "other_scheme"
    sc = _scenario(seeds=[0])
    with pytest.raises(ValueError, match="parcellation_scheme mismatch"):
        backend.generate(sc, ss, np.random.default_rng(0))


def test_parcel_to_vertex_mapping_is_piecewise_constant(tmp_path):
    """Locks in the by-construction property `background = baseline_R[parc, :]`:
    vertices sharing a parcel must carry an identical time series.

    This guarantees that any future refactor that introduces vertex-level
    interpolation, smoothing, or averaging across the parcel boundary will
    break the test rather than silently produce a different ground truth.
    """
    cfg = _tiny_config(tmp_path)
    conn = _tiny_conn()
    backend = TVBSourceGenerator(cfg, conn)
    ss = _tiny_source_space()
    sc = _scenario(seeds=[0])
    _, bg = backend.generate(sc, ss, np.random.default_rng(0))

    parc = ss.parcellation
    for p in np.unique(parc):
        mask = parc == p
        if mask.sum() < 2:
            continue
        ref = bg[mask][0]
        assert np.allclose(bg[mask], ref[None, :], rtol=1e-5, atol=1e-7), (
            f"background varies within parcel {p}"
        )
