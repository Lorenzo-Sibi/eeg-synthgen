from __future__ import annotations

import numpy as np
import pytest

from synthgen.sample import Scenario


def _make_scenario(snr_db: float = 10.0) -> Scenario:
    sc = Scenario(
        scenario_id="test-0",
        seed=0,
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
    sc.snr_db = snr_db
    return sc


def _make_config(tmp_path):
    import yaml
    cfg = {
        "anatomy_bank": {"bank_dir": str(tmp_path / "anatomy")},
        "leadfield_bank": {"bank_dir": str(tmp_path / "leadfield")},
        "montages": {"montages": []},
        "writer": {"output_dir": str(tmp_path / "out")},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    from synthgen.config import GenerationConfig
    return GenerationConfig.from_yaml(p)


# ── SensorNoiseEngine implementations ─────────────────────────────────────────

def test_white_gaussian_output_shape():
    from synthgen.acquisition.noise import WhiteGaussianNoise
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario(snr_db=10.0)
    engine = WhiteGaussianNoise()
    out = engine.apply(clean, sc, rng)
    assert out.shape == clean.shape
    assert out.dtype == np.float32


def test_white_gaussian_adds_noise():
    from synthgen.acquisition.noise import WhiteGaussianNoise
    rng = np.random.default_rng(1)
    clean = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario(snr_db=0.0)  # SNR=0dB → noise RMS ≈ signal RMS
    engine = WhiteGaussianNoise()
    out = engine.apply(clean, sc, rng)
    assert not np.allclose(out, clean)


def test_white_gaussian_snr_approximately_correct():
    from synthgen.acquisition.noise import WhiteGaussianNoise
    rng = np.random.default_rng(2)
    # Strong signal for reliable SNR measurement
    clean = np.ones((4, 1000), dtype=np.float32)
    sc = _make_scenario(snr_db=20.0)
    engine = WhiteGaussianNoise()
    out = engine.apply(clean, sc, rng)
    noise = out - clean
    measured_snr = 20.0 * np.log10(float(np.sqrt(np.mean(clean**2))) / (float(np.sqrt(np.mean(noise**2))) + 1e-10))
    assert abs(measured_snr - 20.0) < 3.0  # within 3dB


def test_colored_1f_output_shape(tmp_path):
    from synthgen.acquisition.noise import Colored1fNoise
    config = _make_config(tmp_path)
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario(snr_db=10.0)
    engine = Colored1fNoise(config)
    out = engine.apply(clean, sc, rng)
    assert out.shape == clean.shape
    assert out.dtype == np.float32


def test_colored_1f_adds_noise(tmp_path):
    from synthgen.acquisition.noise import Colored1fNoise
    config = _make_config(tmp_path)
    rng = np.random.default_rng(3)
    clean = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario(snr_db=0.0)
    engine = Colored1fNoise(config)
    out = engine.apply(clean, sc, rng)
    assert not np.allclose(out, clean)


def test_empirical_resting_output_shape():
    from synthgen.acquisition.noise import EmpiricalRestingNoise
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario(snr_db=10.0)
    engine = EmpiricalRestingNoise()
    out = engine.apply(clean, sc, rng)
    assert out.shape == clean.shape
    assert out.dtype == np.float32


def test_empirical_resting_adds_noise():
    from synthgen.acquisition.noise import EmpiricalRestingNoise
    rng = np.random.default_rng(4)
    clean = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario(snr_db=0.0)
    engine = EmpiricalRestingNoise()
    out = engine.apply(clean, sc, rng)
    assert not np.allclose(out, clean)


def test_colored_1f_snr_approximately_correct(tmp_path):
    from synthgen.acquisition.noise import Colored1fNoise
    config = _make_config(tmp_path)
    rng = np.random.default_rng(10)
    clean = np.ones((4, 1000), dtype=np.float32)
    sc = _make_scenario(snr_db=20.0)
    engine = Colored1fNoise(config)
    out = engine.apply(clean, sc, rng)
    noise = out - clean
    measured_snr = 20.0 * np.log10(float(np.sqrt(np.mean(clean**2))) / (float(np.sqrt(np.mean(noise**2))) + 1e-10))
    assert abs(measured_snr - 20.0) < 3.0


def test_empirical_resting_snr_approximately_correct():
    from synthgen.acquisition.noise import EmpiricalRestingNoise
    rng = np.random.default_rng(11)
    clean = np.ones((4, 1000), dtype=np.float32)
    sc = _make_scenario(snr_db=20.0)
    engine = EmpiricalRestingNoise()
    out = engine.apply(clean, sc, rng)
    noise = out - clean
    measured_snr = 20.0 * np.log10(float(np.sqrt(np.mean(clean**2))) / (float(np.sqrt(np.mean(noise**2))) + 1e-10))
    assert abs(measured_snr - 20.0) < 3.0


def test_empirical_channel_cov_uses_named_channels(tmp_path):
    """With matched channel names the noise should reflect the target cov off-diag."""
    from synthgen.acquisition.noise import EmpiricalChannelCov

    C = 6
    rng = np.random.default_rng(0)
    target = 1e-10 * np.eye(C) + 0.5e-10 * (np.ones((C, C)) - np.eye(C))
    ch_names = [f"C{i}" for i in range(C)]
    bank = tmp_path / "noise.npz"
    np.savez(
        bank,
        sensor_cov=target.astype(np.float32),
        sensor_cov_ch_names=np.array(ch_names, dtype=object),
        _manifest_json=np.array("{}"),
    )

    sc = _make_scenario(snr_db=0.0)
    engine = EmpiricalChannelCov(bank_path=bank)
    clean = rng.standard_normal((C, 4000)).astype(np.float32)
    out = engine.apply(clean, sc, rng, ch_names=ch_names)
    assert out.shape == (C, 4000)
    noise = out - clean
    emp_corr = np.corrcoef(noise)
    off = emp_corr[~np.eye(C, dtype=bool)]
    # Target off-diagonal correlation ~ 0.5; with 4000 samples expect within ±0.15
    assert abs(float(off.mean()) - 0.5) < 0.15


def test_empirical_channel_cov_falls_back_when_no_overlap(tmp_path):
    from synthgen.acquisition.noise import EmpiricalChannelCov

    bank = tmp_path / "noise.npz"
    np.savez(
        bank,
        sensor_cov=np.eye(3, dtype=np.float32),
        sensor_cov_ch_names=np.array(["A", "B", "C"], dtype=object),
        _manifest_json=np.array("{}"),
    )
    sc = _make_scenario(snr_db=10.0)
    engine = EmpiricalChannelCov(bank_path=bank)
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((4, 500)).astype(np.float32)
    out = engine.apply(clean, sc, rng, ch_names=["X", "Y", "Z", "W"])
    assert out.shape == clean.shape
    assert not np.allclose(out, clean)  # fallback still adds noise


def test_empirical_channel_cov_falls_back_when_bank_missing(tmp_path):
    from synthgen.acquisition.noise import EmpiricalChannelCov

    engine = EmpiricalChannelCov(bank_path=tmp_path / "does_not_exist.npz")
    sc = _make_scenario(snr_db=10.0)
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((4, 500)).astype(np.float32)
    out = engine.apply(clean, sc, rng, ch_names=["A", "B", "C", "D"])
    assert out.shape == clean.shape
    assert not np.allclose(out, clean)


# ── ArtifactEngine implementations ────────────────────────────────────────────

def test_ocular_artifact_output_shape(tmp_path):
    from synthgen.acquisition.artifacts import OcularArtifact
    config = _make_config(tmp_path)
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario()
    engine = OcularArtifact(config)
    out = engine.apply(eeg, sc, rng)
    assert out.shape == eeg.shape
    assert out.dtype == np.float32


def test_ocular_artifact_modifies_eeg(tmp_path):
    from synthgen.acquisition.artifacts import OcularArtifact
    config = _make_config(tmp_path)
    rng = np.random.default_rng(1)
    eeg = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario()
    engine = OcularArtifact(config)
    out = engine.apply(eeg, sc, rng)
    assert not np.allclose(out, eeg)


def test_muscular_artifact_output_shape(tmp_path):
    from synthgen.acquisition.artifacts import MuscularArtifact
    config = _make_config(tmp_path)
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario()
    engine = MuscularArtifact(config)
    out = engine.apply(eeg, sc, rng)
    assert out.shape == eeg.shape
    assert out.dtype == np.float32


def test_line_noise_artifact_output_shape(tmp_path):
    from synthgen.acquisition.artifacts import LineNoiseArtifact
    config = _make_config(tmp_path)
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario()
    engine = LineNoiseArtifact(config)
    out = engine.apply(eeg, sc, rng)
    assert out.shape == eeg.shape
    assert out.dtype == np.float32


def test_line_noise_artifact_modifies_eeg(tmp_path):
    from synthgen.acquisition.artifacts import LineNoiseArtifact
    config = _make_config(tmp_path)
    rng = np.random.default_rng(2)
    eeg = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario()
    engine = LineNoiseArtifact(config)
    out = engine.apply(eeg, sc, rng)
    assert not np.allclose(out, eeg)


def test_bad_channel_dropout_zeros_channels():
    from synthgen.acquisition.artifacts import BadChannelDropout
    rng = np.random.default_rng(0)
    C = 16
    eeg = rng.standard_normal((C, 500)).astype(np.float32)
    sc = _make_scenario()
    engine = BadChannelDropout()
    out = engine.apply(eeg, sc, rng)
    assert out.shape == eeg.shape
    # At least one channel should be zeroed out
    zero_channels = np.sum(np.all(out == 0.0, axis=1))
    assert zero_channels >= 1


def test_bad_channel_dropout_output_dtype():
    from synthgen.acquisition.artifacts import BadChannelDropout
    rng = np.random.default_rng(5)
    eeg = rng.standard_normal((8, 500)).astype(np.float32)
    sc = _make_scenario()
    engine = BadChannelDropout()
    out = engine.apply(eeg, sc, rng)
    assert out.dtype == np.float32


# ── QC checker ────────────────────────────────────────────────────────────────

def test_qc_passes_valid_sample(tmp_path):
    from synthgen.acquisition.qc import check_sample, QCResult
    from synthgen.sample import EEGSample

    config = _make_config(tmp_path)
    sc = _make_scenario()
    sc.inter_source_distances_mm = [50.0]
    sc.sir_db = 10.0
    C, T, N = 16, 500, 20
    rng = np.random.default_rng(0)
    sample = EEGSample(
        eeg=rng.standard_normal((C, T)).astype(np.float32),
        source_activity=np.zeros((N, T), dtype=np.float32),
        source_support=np.zeros(N, dtype=bool),
        electrode_coords=np.zeros((C, 3), dtype=np.float32),
        source_coords=np.zeros((N, 3), dtype=np.float32),
        params=sc,
        sir_measured_db=10.0,
        snr_measured_db=15.0,
        sinr_measured_db=8.7,
        active_area_cm2=3.0,
        config_hash="deadbeef",
    )
    result = check_sample(sample, config.qc)
    assert isinstance(result, QCResult)
    assert result.passed


def test_qc_fails_too_few_valid_channels(tmp_path):
    from synthgen.acquisition.qc import check_sample
    from synthgen.sample import EEGSample

    config = _make_config(tmp_path)
    sc = _make_scenario()
    C, T, N = 8, 500, 20  # C=8 < min_valid_channels default of 10
    # Zero-out all channels → no valid channels
    sample = EEGSample(
        eeg=np.zeros((C, T), dtype=np.float32),
        source_activity=np.zeros((N, T), dtype=np.float32),
        source_support=np.zeros(N, dtype=bool),
        electrode_coords=np.zeros((C, 3), dtype=np.float32),
        source_coords=np.zeros((N, 3), dtype=np.float32),
        params=sc,
        sir_measured_db=10.0,
        snr_measured_db=15.0,
        sinr_measured_db=8.7,
        active_area_cm2=3.0,
        config_hash="deadbeef",
    )
    result = check_sample(sample, config.qc)
    assert not result.passed
    assert len(result.reasons) >= 1


def test_qc_fails_sources_too_close(tmp_path):
    from synthgen.acquisition.qc import check_sample
    from synthgen.sample import EEGSample

    config = _make_config(tmp_path)
    sc = _make_scenario()
    sc.inter_source_distances_mm = [2.0]  # below min_inter_source_distance_mm=10
    C, T, N = 16, 500, 20
    rng = np.random.default_rng(0)
    sample = EEGSample(
        eeg=rng.standard_normal((C, T)).astype(np.float32),
        source_activity=np.zeros((N, T), dtype=np.float32),
        source_support=np.zeros(N, dtype=bool),
        electrode_coords=np.zeros((C, 3), dtype=np.float32),
        source_coords=np.zeros((N, 3), dtype=np.float32),
        params=sc,
        sir_measured_db=10.0,
        snr_measured_db=15.0,
        sinr_measured_db=8.7,
        active_area_cm2=3.0,
        config_hash="deadbeef",
    )
    result = check_sample(sample, config.qc)
    assert not result.passed
    assert any("distance" in r for r in result.reasons)


def test_qc_single_source_no_distance_check(tmp_path):
    from synthgen.acquisition.qc import check_sample
    from synthgen.sample import EEGSample

    config = _make_config(tmp_path)
    sc = _make_scenario()
    sc.inter_source_distances_mm = []  # single source → no pairwise distances
    C, T, N = 16, 500, 20
    rng = np.random.default_rng(0)
    sample = EEGSample(
        eeg=rng.standard_normal((C, T)).astype(np.float32),
        source_activity=np.zeros((N, T), dtype=np.float32),
        source_support=np.zeros(N, dtype=bool),
        electrode_coords=np.zeros((C, 3), dtype=np.float32),
        source_coords=np.zeros((N, 3), dtype=np.float32),
        params=sc,
        sir_measured_db=10.0,
        snr_measured_db=15.0,
        sinr_measured_db=8.7,
        active_area_cm2=3.0,
        config_hash="deadbeef",
    )
    result = check_sample(sample, config.qc)
    assert result.passed


# ── AcquisitionPipeline ────────────────────────────────────────────────────────

def _make_pipeline_inputs(N: int = 20, C: int = 8, T: int = 500):
    """Return (source_activity, background_activity, leadfield, electrode_coords, ch_names, source_space)."""
    import scipy.sparse as sp
    from synthgen.sample import SourceSpace

    rng = np.random.default_rng(99)
    G = rng.standard_normal((C, N)).astype(np.float32) * 0.01
    source_activity = np.zeros((N, T), dtype=np.float32)
    source_activity[0] = rng.standard_normal(T).astype(np.float32)
    background_activity = rng.standard_normal((N, T)).astype(np.float32) * 0.1
    electrode_coords = rng.standard_normal((C, 3)).astype(np.float32) * 80.0
    ch_names = [f"EEG{i:03d}" for i in range(C)]
    ss = SourceSpace(
        vertex_coords=rng.standard_normal((N, 3)).astype(np.float32) * 50.0,
        adjacency=sp.eye(N, format="csr", dtype=np.float32),
        parcellation=np.zeros(N, dtype=np.int32),
        hemisphere=np.zeros(N, dtype=np.int32),
    )
    return source_activity, background_activity, G, electrode_coords, ch_names, ss


def test_pipeline_run_returns_eeg_sample(tmp_path):
    from synthgen.acquisition.pipeline import AcquisitionPipeline
    from synthgen.sample import EEGSample
    config = _make_config(tmp_path)
    N, C, T = 20, 8, config.temporal.n_samples_per_window
    src, bg, G, ecords, ch_names, ss = _make_pipeline_inputs(N=N, C=C, T=T)
    sc = _make_scenario()
    sc.seed_vertex_indices = [0]
    sc.dominant_frequencies_hz = [10.0]
    sc.temporal_onsets_s = [0.0]
    pipeline = AcquisitionPipeline(config)
    sample, qc = pipeline.run(sc, ss, src, bg, G, ecords, ch_names, np.random.default_rng(0))
    assert isinstance(sample, EEGSample)
    assert sample.eeg.shape == (C, T)
    assert sample.source_activity.shape == (N, T)
    assert sample.electrode_coords.shape == (C, 3)
    assert sample.source_coords.shape == (N, 3)


def test_pipeline_run_eeg_dtype(tmp_path):
    from synthgen.acquisition.pipeline import AcquisitionPipeline
    config = _make_config(tmp_path)
    N, C, T = 20, 8, config.temporal.n_samples_per_window
    src, bg, G, ecords, ch_names, ss = _make_pipeline_inputs(N=N, C=C, T=T)
    sc = _make_scenario()
    sc.seed_vertex_indices = [0]
    sc.dominant_frequencies_hz = [10.0]
    sc.temporal_onsets_s = [0.0]
    pipeline = AcquisitionPipeline(config)
    sample, _ = pipeline.run(sc, ss, src, bg, G, ecords, ch_names, np.random.default_rng(1))
    assert sample.eeg.dtype == np.float32


def test_pipeline_run_fills_snir_on_scenario(tmp_path):
    from synthgen.acquisition.pipeline import AcquisitionPipeline
    config = _make_config(tmp_path)
    N, C, T = 20, 8, config.temporal.n_samples_per_window
    src, bg, G, ecords, ch_names, ss = _make_pipeline_inputs(N=N, C=C, T=T)
    sc = _make_scenario()
    sc.seed_vertex_indices = [0]
    sc.dominant_frequencies_hz = [10.0]
    sc.temporal_onsets_s = [0.0]
    pipeline = AcquisitionPipeline(config)
    sample, _ = pipeline.run(sc, ss, src, bg, G, ecords, ch_names, np.random.default_rng(2))
    # The pipeline must draw both SNIR and sensor SNR from the configured grids.
    assert sc.sir_db in config.noise.sir_levels_db
    assert sc.snr_db in config.noise.snr_levels_db


def test_pipeline_run_source_support_marks_seeds(tmp_path):
    from synthgen.acquisition.pipeline import AcquisitionPipeline
    config = _make_config(tmp_path)
    N, C, T = 20, 8, config.temporal.n_samples_per_window
    src, bg, G, ecords, ch_names, ss = _make_pipeline_inputs(N=N, C=C, T=T)
    sc = _make_scenario()
    sc.seed_vertex_indices = [0, 5]
    sc.dominant_frequencies_hz = [10.0, 10.0]
    sc.temporal_onsets_s = [0.0, 0.0]
    pipeline = AcquisitionPipeline(config)
    sample, _ = pipeline.run(sc, ss, src, bg, G, ecords, ch_names, np.random.default_rng(3))
    assert sample.source_support[0]
    assert sample.source_support[5]
    assert not sample.source_support[1]


def test_pipeline_run_reproducible(tmp_path):
    from synthgen.acquisition.pipeline import AcquisitionPipeline
    config = _make_config(tmp_path)
    N, C, T = 20, 8, config.temporal.n_samples_per_window
    src, bg, G, ecords, ch_names, ss = _make_pipeline_inputs(N=N, C=C, T=T)
    pipeline = AcquisitionPipeline(config)

    def _make_sc():
        s = _make_scenario()
        s.seed_vertex_indices = [0]
        s.dominant_frequencies_hz = [10.0]
        s.temporal_onsets_s = [0.0]
        return s

    sample1, _ = pipeline.run(_make_sc(), ss, src, bg, G, ecords, ch_names, np.random.default_rng(77))
    sample2, _ = pipeline.run(_make_sc(), ss, src, bg, G, ecords, ch_names, np.random.default_rng(77))
    np.testing.assert_array_equal(sample1.eeg, sample2.eeg)


def test_pipeline_config_hash_in_sample(tmp_path):
    from synthgen.acquisition.pipeline import AcquisitionPipeline
    config = _make_config(tmp_path)
    N, C, T = 20, 8, config.temporal.n_samples_per_window
    src, bg, G, ecords, ch_names, ss = _make_pipeline_inputs(N=N, C=C, T=T)
    sc = _make_scenario()
    sc.seed_vertex_indices = [0]
    sc.dominant_frequencies_hz = [10.0]
    sc.temporal_onsets_s = [0.0]
    pipeline = AcquisitionPipeline(config)
    sample, _ = pipeline.run(sc, ss, src, bg, G, ecords, ch_names, np.random.default_rng(4))
    assert isinstance(sample.config_hash, str)
    assert len(sample.config_hash) > 0
