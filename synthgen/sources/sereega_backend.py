from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from synthgen.config import GenerationConfig
from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.base import SourceGeneratorBackend


def _erp(T: int, sfreq: float, onset_s: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(T) / sfreq
    peak_t = onset_s + float(rng.uniform(0.05, 0.2))
    width = float(rng.uniform(0.02, 0.08))
    amplitude = float(rng.uniform(0.5, 2.0))
    return (amplitude * np.exp(-0.5 * ((t - peak_t) / width) ** 2)).astype(np.float32)


def _oscillatory_burst(
    T: int, sfreq: float, freq_hz: float, onset_s: float, rng: np.random.Generator
) -> np.ndarray:
    t = np.arange(T) / sfreq
    burst_center = onset_s + float(rng.uniform(0.1, 0.3))
    burst_width = float(rng.uniform(0.1, 0.2))
    amplitude = float(rng.uniform(0.5, 2.0))
    phase = float(rng.uniform(0.0, 2 * np.pi))
    envelope = np.exp(-0.5 * ((t - burst_center) / burst_width) ** 2)
    return (amplitude * envelope * np.sin(2 * np.pi * freq_hz * t + phase)).astype(np.float32)


def _ar_correlated(T: int, rng: np.random.Generator) -> np.ndarray:
    ar_coef = float(rng.uniform(0.7, 0.97))
    noise = rng.standard_normal(T).astype(np.float64)
    sig = lfilter([1.0], [1.0, -ar_coef], noise).astype(np.float32)
    std = float(np.std(sig))
    return sig / (std + 1e-8)


def _spike_interictal(T: int, sfreq: float, onset_s: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(T) / sfreq
    spike_t = onset_s + float(rng.uniform(0.05, 0.3))
    wave_t = spike_t + 0.04
    amplitude = float(rng.uniform(1.0, 3.0))
    sig = (
        -amplitude * np.exp(-0.5 * ((t - spike_t) / 0.008) ** 2)
        + 0.5 * amplitude * np.exp(-0.5 * ((t - wave_t) / 0.06) ** 2)
    )
    return sig.astype(np.float32)


def _generate_signal(
    signal_family: str,
    T: int,
    sfreq: float,
    freq_hz: float,
    onset_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if signal_family == "erp":
        return _erp(T, sfreq, onset_s, rng)
    if signal_family == "oscillatory_burst":
        return _oscillatory_burst(T, sfreq, freq_hz, onset_s, rng)
    if signal_family == "ar_correlated":
        return _ar_correlated(T, rng)
    if signal_family == "spike_interictal":
        return _spike_interictal(T, sfreq, onset_s, rng)
    raise ValueError(f"Unknown signal family: {signal_family!r}")


def _generate_1f_background(N: int, T: int, sfreq: float, rng: np.random.Generator) -> np.ndarray:
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    freqs[0] = 1.0  # avoid division by zero at DC
    phases = rng.uniform(0.0, 2 * np.pi, size=(N, len(freqs)))
    amplitudes = 1.0 / np.sqrt(freqs)
    fft_vals = amplitudes * np.exp(1j * phases)
    bg = np.fft.irfft(fft_vals, n=T).astype(np.float32)
    std = np.std(bg, axis=1, keepdims=True)
    return (bg / (std + 1e-8) * 0.1).astype(np.float32)


class SEREEGABackend(SourceGeneratorBackend):
    """Source generator using SEREEGA-style patch-based modeling.

    Signal waveforms (ERP, oscillatory burst, AR, spike) are generated
    in Python. MATLAB engine integration for the full SEREEGA toolbox
    is deferred to a future release; the generate() interface is stable.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config

    def generate(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        T = self._config.temporal.n_samples_per_window
        N = len(source_space.vertex_coords)
        sfreq = self._config.temporal.sfreq

        n_seeds = len(scenario.seed_vertex_indices)
        if len(scenario.dominant_frequencies_hz) < n_seeds:
            raise ValueError(
                f"dominant_frequencies_hz has {len(scenario.dominant_frequencies_hz)} entries "
                f"but seed_vertex_indices has {n_seeds}"
            )
        if len(scenario.temporal_onsets_s) < n_seeds:
            raise ValueError(
                f"temporal_onsets_s has {len(scenario.temporal_onsets_s)} entries "
                f"but seed_vertex_indices has {n_seeds}"
            )

        source_activity = np.zeros((N, T), dtype=np.float32)
        patches = scenario.seed_patch_vertex_indices
        for i, seed in enumerate(scenario.seed_vertex_indices):
            freq = scenario.dominant_frequencies_hz[i]
            onset = scenario.temporal_onsets_s[i]
            waveform = _generate_signal(scenario.signal_family, T, sfreq, freq, onset, rng)
            patch = patches[i] if i < len(patches) and patches[i] else [seed]
            source_activity[patch] = waveform

        background_activity = _generate_1f_background(N, T, sfreq, rng)
        return source_activity, background_activity
