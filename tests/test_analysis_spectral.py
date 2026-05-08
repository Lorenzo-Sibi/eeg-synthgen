import numpy as np
import pytest

from synthgen.analysis.spectral import (
    EEG_BANDS,
    NORMATIVE_BACKGROUND_SLOPE_RANGE,
    compute_band_powers,
    compute_psd,
    estimate_psd_slope,
)

SFREQ, T = 500.0, 2000


def _sine(f):
    return np.sin(2 * np.pi * f * np.arange(T) / SFREQ).astype(np.float32)


def _pink(beta=1.0, seed=0):
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(T, 1 / SFREQ)
    freqs[0] = 1.0
    fft = np.fft.rfft(rng.standard_normal(T)) / np.power(freqs, beta / 2)
    return np.fft.irfft(fft, n=T).astype(np.float32)


@pytest.mark.parametrize(
    "freq,expected_band",
    [(2.0, "delta"), (10.0, "alpha"), (20.0, "beta")],
)
def test_band_dominant(freq, expected_band):
    bp = compute_band_powers(_sine(freq), SFREQ)
    assert max(bp, key=bp.get) == expected_band


@pytest.mark.parametrize(
    "trace_fn,low,high",
    [
        (
            lambda: np.random.default_rng(1).standard_normal(T).astype(np.float32),
            -0.5,
            0.5,
        ),
        (lambda: _pink(1.0), 0.5, 1.5),
        (lambda: (_pink(1.0) + 0.6 * _sine(10.0)).astype(np.float32), 0.5, 1.5),
    ],
)
def test_slope_in_range(trace_fn, low, high):
    assert low < estimate_psd_slope(trace_fn(), SFREQ) < high


def test_slope_nan_when_too_short():
    assert np.isnan(estimate_psd_slope(np.zeros(8, np.float32), SFREQ))


def test_normative_range():
    lo, hi = NORMATIVE_BACKGROUND_SLOPE_RANGE
    assert 0.5 < lo < hi < 3.0


def test_compute_psd_shapes():
    freqs, psd = compute_psd(_pink(), SFREQ)
    assert freqs.size == psd.size
    assert np.isfinite(psd).all()


def test_band_powers_keys_match():
    bp = compute_band_powers(_pink(), SFREQ)
    assert set(bp.keys()) == set(EEG_BANDS.keys())
