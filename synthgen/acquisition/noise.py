from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from synthgen.sample import Scenario


class SensorNoiseEngine(ABC):
    @abstractmethod
    def apply(
        self,
        clean_eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add sensor noise to clean_eeg. Returns noisy_eeg: CxT."""


class WhiteGaussianNoise(SensorNoiseEngine):
    def apply(
        self,
        clean_eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        snr_linear = 10.0 ** (scenario.snr_sensor_db / 20.0)
        signal_rms = float(np.sqrt(np.mean(clean_eeg ** 2)))
        noise_std = signal_rms / (snr_linear + 1e-10)
        noise = rng.standard_normal(clean_eeg.shape).astype(np.float32) * noise_std
        return (clean_eeg + noise).astype(np.float32)


class Colored1fNoise(SensorNoiseEngine):
    def __init__(self, config) -> None:
        self._sfreq = config.temporal.sfreq

    def apply(
        self,
        clean_eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        C, T = clean_eeg.shape
        snr_linear = 10.0 ** (scenario.snr_sensor_db / 20.0)
        signal_rms = float(np.sqrt(np.mean(clean_eeg ** 2)))
        target_noise_rms = signal_rms / (snr_linear + 1e-10)

        freqs = np.fft.rfftfreq(T, d=1.0 / self._sfreq)
        freqs[0] = 1.0  # avoid DC divide-by-zero
        amplitudes = 1.0 / np.sqrt(freqs)
        phases = rng.uniform(0.0, 2 * np.pi, size=(C, len(freqs)))
        fft_vals = amplitudes * np.exp(1j * phases)
        noise = np.fft.irfft(fft_vals, n=T).astype(np.float32)

        noise_rms = float(np.sqrt(np.mean(noise ** 2)))
        noise = noise / (noise_rms + 1e-10) * target_noise_rms
        return (clean_eeg + noise).astype(np.float32)


class EmpiricalRestingNoise(SensorNoiseEngine):
    """AR(2) per-channel colored noise approximating resting-state sensor noise."""

    def apply(
        self,
        clean_eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        from scipy.signal import lfilter

        C, T = clean_eeg.shape
        snr_linear = 10.0 ** (scenario.snr_sensor_db / 20.0)
        signal_rms = float(np.sqrt(np.mean(clean_eeg ** 2)))
        target_noise_rms = signal_rms / (snr_linear + 1e-10)

        noise = np.zeros((C, T), dtype=np.float32)
        for c in range(C):
            ar1 = float(rng.uniform(0.6, 0.9))
            ar2 = float(rng.uniform(-0.3, 0.0))
            white = rng.standard_normal(T).astype(np.float64)
            sig = lfilter([1.0], [1.0, -ar1, -ar2], white).astype(np.float32)
            std = float(np.std(sig))
            noise[c] = sig / (std + 1e-8)

        noise_rms = float(np.sqrt(np.mean(noise ** 2)))
        noise = noise / (noise_rms + 1e-10) * target_noise_rms
        return (clean_eeg + noise).astype(np.float32)
