import numpy as np
import pytest
import scipy.sparse as sp
from dataclasses import asdict

from synthgen.sample import EEGSample, Scenario, SourceSpace


def _make_scenario(**kwargs) -> Scenario:
    defaults = dict(
        scenario_id="test-001",
        seed=123    ,
        anatomy_id="fsaverage",
        leadfield_id="fsaverage_64ch_standard",
        montage_id="standard_1005_64",
        reference_scheme="average",
        conductivity_id="standard",
        prior_family="local_contiguous",
        n_sources=1,
        signal_family="erp",
        split="train",
    )
    defaults.update(kwargs)
    return Scenario(**defaults)


def test_scenario_required_fields():
    s = _make_scenario()
    assert s.scenario_id == "test-001"
    assert s.n_sources == 1


def test_scenario_spatial_params_default_empty():
    s = _make_scenario()
    assert s.patch_extents_cm2 == []
    assert s.seed_vertex_indices == []
    assert s.inter_source_distances_mm == []
    assert s.temporal_onsets_s == []
    assert s.dominant_frequencies_hz == []


def test_scenario_noise_params_default_zero():
    s = _make_scenario()
    assert s.snir_db == 0.0
    assert s.snr_sensor_db == 0.0
    assert s.artifact_flags == []


def test_scenario_independent_default_lists():
    s1 = _make_scenario()
    s2 = _make_scenario()
    s1.patch_extents_cm2.append(5.0)
    assert s2.patch_extents_cm2 == []


def test_source_space_shapes():
    N = 200
    ss = SourceSpace(
        vertex_coords=np.zeros((N, 3)),
        adjacency=sp.eye(N, format="csr"),
        parcellation=np.zeros(N, dtype=int),
        hemisphere=np.zeros(N, dtype=int),
    )
    assert ss.vertex_coords.shape == (N, 3)
    assert ss.adjacency.shape == (N, N)
    assert ss.parcellation.shape == (N,)
    assert ss.hemisphere.shape == (N,)


def test_eeg_sample_shapes():
    C, T, N = 64, 500, 2004
    s = _make_scenario()
    sample = EEGSample(
        eeg=np.zeros((C, T)),
        source_activity=np.zeros((N, T)),
        source_support=np.zeros(N, dtype=bool),
        electrode_coords=np.zeros((C, 3)),
        source_coords=np.zeros((N, 3)),
        params=s,
        snir_measured_db=15.0,
        snr_sensor_measured_db=10.0,
        active_area_cm2=5.0,
        config_hash="deadbeef",
    )
    assert sample.eeg.shape == (C, T)
    assert sample.source_activity.shape == (N, T)
    assert sample.source_support.shape == (N,)
    assert sample.electrode_coords.shape == (C, 3)
    assert sample.source_coords.shape == (N, 3)


def test_scenario_is_serializable():
    s = _make_scenario()
    d = asdict(s)
    assert isinstance(d, dict)
    assert d["scenario_id"] == "test-001"
