from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from synthgen.sample import Scenario


class SensorNoiseEngine(ABC):
    @abstractmethod
    def apply(
        self,
        clean_eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
        ch_names: list[str] | None = None,
    ) -> np.ndarray:
        """Add sensor noise to clean_eeg. Returns noisy_eeg: CxT.

        ``ch_names`` is the channel labels of ``clean_eeg`` rows. Engines that
        do not depend on channel identity (white, 1/f, empirical AR) may ignore
        it; engines that read a calibrated covariance bank use it for
        name-matching.
        """


class WhiteGaussianNoise(SensorNoiseEngine):
    def apply(
        self,
        clean_eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
        ch_names: list[str] | None = None,
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
        ch_names: list[str] | None = None,
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
        ch_names: list[str] | None = None,
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


class EmpiricalChannelCov(SensorNoiseEngine):
    """N(0, Σ_C) sensor noise, Σ_C calibrated from real EEG.

    The noise bank (e.g. ``banks/noise/physionet_v1.npz``) stores one
    covariance matrix together with the channel names it was estimated on.
    At ``apply`` time the engine matches ``ch_names`` to the bank's names
    and uses the corresponding sub-covariance; unmatched channels get
    independent white Gaussian noise at the same target RMS. If the bank
    is missing or no channel overlaps, the engine falls back to fully
    white Gaussian noise.
    """

    def __init__(self, bank_path) -> None:
        self._bank_path = Path(bank_path)
        self._bank = np.load(self._bank_path, allow_pickle=True) if self._bank_path.exists() else None
        if self._bank is not None and "sensor_cov_ch_names" in self._bank.files:
            self._bank_ch_names = [str(c) for c in self._bank["sensor_cov_ch_names"]]
            self._bank_cov = self._bank["sensor_cov"].astype(np.float64)
        else:
            self._bank_ch_names = []
            self._bank_cov = None
        self._chol_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}

    def _build_chol(self, ch_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (chol_factor (n_matched, n_matched), matched_rows (n_matched,))."""
        bank_idx = {c: i for i, c in enumerate(self._bank_ch_names)}
        matched_rows = [i for i, c in enumerate(ch_names) if c in bank_idx]
        bank_rows = [bank_idx[ch_names[i]] for i in matched_rows]
        sub = self._bank_cov[np.ix_(bank_rows, bank_rows)]
        jitter = 1e-12 * (float(np.trace(sub)) / max(sub.shape[0], 1) + 1e-30)
        chol = np.linalg.cholesky(sub + jitter * np.eye(sub.shape[0]))
        return chol, np.array(matched_rows, dtype=np.int64)

    def apply(
        self,
        clean_eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
        ch_names: list[str] | None = None,
    ) -> np.ndarray:
        C, T = clean_eeg.shape
        snr_linear = 10.0 ** (scenario.snr_sensor_db / 20.0)
        signal_rms = float(np.sqrt(np.mean(clean_eeg ** 2)))
        target_noise_rms = signal_rms / (snr_linear + 1e-10)

        if self._bank_cov is None or not ch_names:
            noise = rng.standard_normal((C, T)).astype(np.float32) * target_noise_rms
            return (clean_eeg + noise).astype(np.float32)

        key = tuple(ch_names)
        if key not in self._chol_cache:
            self._chol_cache[key] = self._build_chol(list(ch_names))
        chol, matched_rows = self._chol_cache[key]

        noise = rng.standard_normal((C, T)).astype(np.float64)
        if matched_rows.size > 0:
            white_matched = rng.standard_normal((matched_rows.size, T))
            noise[matched_rows] = chol @ white_matched
        noise_rms = float(np.sqrt(np.mean(noise ** 2)))
        noise = noise / (noise_rms + 1e-10) * target_noise_rms
        return (clean_eeg + noise.astype(np.float32)).astype(np.float32)
