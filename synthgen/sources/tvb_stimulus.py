from __future__ import annotations

import numpy as np

from synthgen.sample import Scenario


def seed_parcel_ids(scenario: Scenario, parcellation: np.ndarray) -> np.ndarray:
    """Return unique parcel ids touched by the scenario's seed patches.

    Falls back to ``seed_vertex_indices`` when ``seed_patch_vertex_indices`` is
    empty (e.g. for legacy scenarios or tests that construct Scenarios by hand).
    """
    patches = scenario.seed_patch_vertex_indices
    if patches:
        flat: list[int] = []
        for p in patches:
            flat.extend(int(v) for v in p)
        if not flat:
            return np.array([], dtype=np.int64)
        ids = np.unique(parcellation[np.asarray(flat, dtype=np.int64)])
        return ids.astype(np.int64)
    if not scenario.seed_vertex_indices:
        return np.array([], dtype=np.int64)
    ids = np.unique(parcellation[np.asarray(scenario.seed_vertex_indices, dtype=np.int64)])
    return ids.astype(np.int64)


def _erp_pattern(T: int, sfreq: float, onset_s: float, amp: float) -> np.ndarray:
    t = np.arange(T) / sfreq
    width = 0.05
    peak = onset_s + 0.12
    return (amp * np.exp(-0.5 * ((t - peak) / width) ** 2)).astype(np.float32)


def _burst_pattern(T: int, sfreq: float, onset_s: float, freq_hz: float, amp: float) -> np.ndarray:
    t = np.arange(T) / sfreq
    center = onset_s + 0.2
    width = 0.15
    env = np.exp(-0.5 * ((t - center) / width) ** 2)
    return (amp * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _ar_pattern(T: int, rng: np.random.Generator, amp: float) -> np.ndarray:
    from scipy.signal import lfilter

    x = lfilter([1.0], [1.0, -0.9], rng.standard_normal(T)).astype(np.float32)
    x = x / (x.std() + 1e-8)
    return (amp * x).astype(np.float32)


def _spike_pattern(T: int, sfreq: float, onset_s: float, amp: float) -> np.ndarray:
    t = np.arange(T) / sfreq
    spike_t = onset_s + 0.1
    wave_t = spike_t + 0.04
    sig = (
        -amp * np.exp(-0.5 * ((t - spike_t) / 0.008) ** 2)
        + 0.5 * amp * np.exp(-0.5 * ((t - wave_t) / 0.06) ** 2)
    )
    return sig.astype(np.float32)


def build_stimulus_pattern(
    scenario: Scenario,
    T: int,
    sfreq: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a (T,) temporal stimulus pattern shared across stimulated regions."""
    family = scenario.signal_family
    onset = scenario.temporal_onsets_s[0] if scenario.temporal_onsets_s else 0.0
    freq = scenario.dominant_frequencies_hz[0] if scenario.dominant_frequencies_hz else 10.0
    if family == "erp":
        return _erp_pattern(T, sfreq, onset, amplitude)
    if family == "oscillatory_burst":
        return _burst_pattern(T, sfreq, onset, freq, amplitude)
    if family == "ar_correlated":
        return _ar_pattern(T, rng, amplitude)
    if family == "spike_interictal":
        return _spike_pattern(T, sfreq, onset, amplitude)
    raise ValueError(f"Unknown signal family: {family!r}")