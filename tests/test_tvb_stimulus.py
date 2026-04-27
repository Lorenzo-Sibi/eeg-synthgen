import numpy as np
import pytest

from synthgen.sample import Scenario
from synthgen.sources.tvb_stimulus import build_stimulus_pattern, seed_parcel_ids


def _scenario(family, seeds=None, onsets=(0.1,), freqs=(10.0,)):
    if seeds is None:
        seeds = [0]
    return Scenario(
        scenario_id="s", seed=0, anatomy_id="a", leadfield_id="l",
        montage_id="m", reference_scheme="average", conductivity_id="standard",
        prior_family="broad_random", n_sources=1, signal_family=family,
        difficulty="easy", split="train",
        seed_vertex_indices=list(seeds),
        temporal_onsets_s=list(onsets),
        dominant_frequencies_hz=list(freqs),
    )


@pytest.mark.parametrize("family", ["erp", "oscillatory_burst", "ar_correlated", "spike_interictal"])
def test_pattern_shape_and_finite(family):
    rng = np.random.default_rng(0)
    p = build_stimulus_pattern(_scenario(family), T=500, sfreq=500.0, amplitude=1.0, rng=rng)
    assert p.shape == (500,)
    assert p.dtype == np.float32
    assert np.isfinite(p).all()


def test_unknown_family_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        build_stimulus_pattern(_scenario("bogus"), T=10, sfreq=500.0, amplitude=1.0, rng=rng)


def test_seed_parcel_ids_unique_and_sorted():
    parc = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    sc = _scenario("erp", seeds=[0, 5, 7])
    ids = seed_parcel_ids(sc, parc)
    assert ids.tolist() == [0, 2, 3]


def test_seed_parcel_ids_empty():
    parc = np.zeros(5, dtype=np.int32)
    sc = _scenario("erp", seeds=[])
    assert seed_parcel_ids(sc, parc).size == 0


def test_seed_parcel_ids_uses_patches_when_available():
    """When seed_patch_vertex_indices is populated, all parcels touched by any
    patch vertex should appear in the output — not just the parcels at the
    individual seed vertices."""
    parc = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    sc = _scenario("erp", seeds=[0])
    sc.seed_patch_vertex_indices = [[0, 1, 2, 3, 4]]  # touches parcels 0, 1, 2
    ids = seed_parcel_ids(sc, parc)
    assert ids.tolist() == [0, 1, 2]
