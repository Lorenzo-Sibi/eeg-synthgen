from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from synthgen.config import GenerationConfig
from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.sereega_backend import SEREEGABackend


class FakeMatlabBridge:
    def __init__(self, _sereega_path):
        pass

    def generate_signal(self, signal_class, T: int, sfreq: float) -> np.ndarray:
        params = signal_class["params"]
        t = np.arange(T, dtype=np.float64) / sfreq
        if signal_class["type"] == "erp":
            sig = np.zeros(T, dtype=np.float64)
            for latency_ms, width_ms, amplitude in zip(
                params["peak_latencies_ms"],
                params["peak_widths_ms"],
                params["peak_amplitudes"],
            ):
                latency_s = float(latency_ms) / 1000.0
                width_s = float(width_ms) / 1000.0
                sig += float(amplitude) * np.exp(-0.5 * ((t - latency_s) / width_s) ** 2)
            return sig.astype(np.float32)
        if signal_class["type"] == "ersp":
            center_s = float(params["mod_latency_ms"]) / 1000.0
            width_s = float(params["mod_width_ms"]) / 1000.0
            envelope = np.exp(-0.5 * ((t - center_s) / width_s) ** 2)
            phase = 2.0 * np.pi * float(params["phase_cycles"])
            return (
                float(params["amplitude"])
                * envelope
                * np.sin(2.0 * np.pi * float(params["frequency_hz"]) * t + phase)
            ).astype(np.float32)
        rng = np.random.default_rng(int(params["seed"]))
        sig = rng.standard_normal(T).astype(np.float32)
        sig = sig / (float(np.std(sig)) + 1e-8)
        return (float(params["amplitude"]) * sig).astype(np.float32)


@pytest.fixture(autouse=True)
def _fake_matlab_bridge(monkeypatch):
    import synthgen.sources.sereega_backend as sereega_backend

    monkeypatch.setattr(sereega_backend, "MatlabSereegaBridge", FakeMatlabBridge)


def _make_config(tmp_path) -> GenerationConfig:
    import yaml
    sereega_path = tmp_path / "SEREEGA"
    sereega_path.mkdir(exist_ok=True)
    cfg = {
        "anatomy_bank": {"bank_dir": str(tmp_path / "anatomy")},
        "leadfield_bank": {"bank_dir": str(tmp_path / "leadfield")},
        "montages": {"montages": []},
        "writer": {"output_dir": str(tmp_path / "out")},
        "sereega": {"matlab_sereega_path": str(sereega_path)},
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
    config.sereega.patch_spatial_profile = "uniform"
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


def test_sereega_backend_builds_component_per_seed():
    from synthgen.sources.sereega_backend import _scenario_components

    sc = _make_scenario(n_sources=2, signal_family="oscillatory_burst")
    sc.seed_patch_vertex_indices = [[0, 10, 11], [1, 20]]

    components = _scenario_components(sc)

    assert len(components) == 2
    assert components[0]["source"] == 0
    assert components[0]["patch"] == [0, 10, 11]
    assert components[0]["signal_classes"][0]["type"] == "ersp"
    assert components[0]["signal_classes"][0]["params"]["frequency_hz"] == 10.0


def test_sereega_backend_sums_signal_classes():
    from synthgen.sources.sereega_backend import (
        _generate_component_activation,
    )

    T = 500
    sfreq = 500.0
    signal_classes = (
        {
            "type": "erp",
            "params": {
                "peak_latencies_ms": [100.0],
                "peak_widths_ms": [20.0],
                "peak_amplitudes": [1.0],
                "seed": 1,
            },
        },
        {
            "type": "erp",
            "params": {
                "peak_latencies_ms": [200.0],
                "peak_widths_ms": [30.0],
                "peak_amplitudes": [0.5],
                "seed": 2,
            },
        },
    )
    component = {"source": 0, "patch": [0], "signal_classes": signal_classes}

    bridge = FakeMatlabBridge(None)
    combined = _generate_component_activation(component, T, sfreq, bridge)
    manual = sum(
        (bridge.generate_signal(signal_class, T, sfreq) for signal_class in signal_classes),
        start=np.zeros(T, dtype=np.float32),
    )

    np.testing.assert_array_equal(combined, manual.astype(np.float32))


def test_sereega_backend_sums_overlapping_components(tmp_path):
    from synthgen.sources.sereega_backend import (
        SEREEGABackend,
        _generate_component_activation,
        _scenario_components,
    )

    config = _make_config(tmp_path)
    config.sereega.patch_spatial_profile = "uniform"
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=20)

    sc_overlap = _make_scenario(n_sources=2, signal_family="erp")
    sc_overlap.seed_patch_vertex_indices = [[0, 5], [1, 5]]
    src_overlap, _ = backend.generate(sc_overlap, ss, np.random.default_rng(9))

    rng_manual = np.random.default_rng(9)
    components = _scenario_components(sc_overlap, config.sereega, rng_manual)
    act0 = _generate_component_activation(
        components[0],
        config.temporal.n_samples_per_window,
        config.temporal.sfreq,
        FakeMatlabBridge(None),
    )
    act1 = _generate_component_activation(
        components[1],
        config.temporal.n_samples_per_window,
        config.temporal.sfreq,
        FakeMatlabBridge(None),
    )

    np.testing.assert_allclose(src_overlap[5], act0 + act1, atol=1e-6)


def test_sereega_backend_emits_canonical_amplitude(tmp_path):
    from synthgen.sources.sereega_backend import (
        SEREEGABackend,
        _CANONICAL_AMPLITUDE_UV,
    )

    config = _make_config(tmp_path)
    config.sereega.erp_peak_count_weights = {1: 1.0}
    config.sereega.latency_jitter_s_range = (0.10, 0.10)
    config.sereega.erp_width_s_range = (0.05, 0.05)

    backend = SEREEGABackend(config)
    ss = _make_source_space(N=20)
    sc = _make_scenario(n_sources=1, signal_family="erp")
    src, _ = backend.generate(sc, ss, np.random.default_rng(0))

    peak_idx = int(0.10 * config.temporal.sfreq)
    assert src[0, peak_idx] == pytest.approx(_CANONICAL_AMPLITUDE_UV, abs=1e-6)


def test_sereega_backend_emits_canonical_background_amplitude(tmp_path):
    from synthgen.sources.sereega_backend import (
        SEREEGABackend,
        _CANONICAL_BACKGROUND_AMPLITUDE_UV,
    )

    config = _make_config(tmp_path)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=20)
    sc = _make_scenario(n_sources=1)
    _, bg = backend.generate(sc, ss, np.random.default_rng(0))

    np.testing.assert_allclose(
        np.std(bg, axis=1), _CANONICAL_BACKGROUND_AMPLITUDE_UV, atol=1e-5
    )


def test_sereega_backend_records_trial_by_trial_parameters(tmp_path):
    config = _make_config(tmp_path)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=20)
    sc = _make_scenario(n_sources=1, signal_family="oscillatory_burst")

    backend.generate(sc, ss, np.random.default_rng(0))

    params = sc.sereega_trial_parameters[0]
    assert params["source"] == 0
    assert params["patch_size"] == 1
    assert params["signal_classes"][0]["type"] == "ersp"
    assert "mod_latency_ms" in params["signal_classes"][0]["params"]
    assert params["spatial"]["profile"] == "gaussian"


def test_sereega_backend_gaussian_patch_weights_decay_with_distance(tmp_path):
    config = _make_config(tmp_path)
    config.sereega.patch_spatial_profile = "gaussian"
    config.sereega.patch_spatial_sigma_mm_range = (10.0, 10.0)
    config.sereega.erp_peak_count_weights = {1: 1.0}
    config.sereega.latency_jitter_s_range = (0.10, 0.10)
    config.sereega.erp_width_s_range = (0.02, 0.02)
    backend = SEREEGABackend(config)
    ss = _make_source_space(N=3)
    ss.vertex_coords = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    sc = _make_scenario(n_sources=1, signal_family="erp")
    sc.seed_patch_vertex_indices = [[0, 1, 2]]

    src, _ = backend.generate(sc, ss, np.random.default_rng(0))

    peak_idx = int(0.10 * config.temporal.sfreq)
    assert abs(src[0, peak_idx]) > abs(src[1, peak_idx]) > abs(src[2, peak_idx])
    spatial = sc.sereega_trial_parameters[0]["spatial"]
    assert spatial["sigma_mm"] == pytest.approx(10.0)
    assert spatial["weight_max"] == pytest.approx(1.0)


def test_sereega_backend_requires_matlab_path(tmp_path):
    config = _make_config(tmp_path)
    config.sereega.matlab_sereega_path = None

    with pytest.raises(RuntimeError, match="matlab_sereega_path"):
        SEREEGABackend(config)
