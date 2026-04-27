import numpy as np
import pytest

from synthgen.sources.tvb_reservoir import StateReservoir


class _FakeSim:
    def __init__(self, total_samples=1000, state_dim=(2, 4, 1)):
        self.total_samples = total_samples
        self.state_dim = state_dim
        self.simulation_length = 0.0
        self.configured = False

    def configure(self):
        self.configured = True

    def run(self):
        n = self.total_samples
        t = np.linspace(0.0, self.simulation_length, n, dtype=np.float64)
        s = np.tile(np.arange(n, dtype=np.float32)[:, None, None, None], (1,) + self.state_dim)
        return ((t, s),)


def test_reservoir_samples_count():
    sim = _FakeSim(total_samples=1000)
    r = StateReservoir(sim, warmup_s=0.2, reservoir_duration_s=0.8, n_states=10)
    assert len(r) == 10


def test_reservoir_sample_shape():
    sim = _FakeSim(total_samples=1000)
    r = StateReservoir(sim, warmup_s=0.2, reservoir_duration_s=0.8, n_states=10)
    rng = np.random.default_rng(0)
    s = r.sample(rng)
    assert s.shape == (2, 4, 1)


def test_reservoir_insufficient_samples_raises():
    sim = _FakeSim(total_samples=10)
    with pytest.raises(ValueError):
        StateReservoir(sim, warmup_s=0.1, reservoir_duration_s=0.9, n_states=100)


def test_reservoir_determinism():
    sim = _FakeSim(total_samples=1000)
    r = StateReservoir(sim, warmup_s=0.2, reservoir_duration_s=0.8, n_states=10)
    a = r.sample(np.random.default_rng(42))
    b = r.sample(np.random.default_rng(42))
    assert np.array_equal(a, b)
