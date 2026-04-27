from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from synthgen.sample import Scenario


class ArtifactEngine(ABC):
    @abstractmethod
    def apply(
        self,
        eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Inject artifact into eeg (CxT). Returns modified eeg: CxT float32."""


class OcularArtifact(ArtifactEngine):
    """Gaussian-envelope blink template with random per-channel amplitude."""

    def __init__(self, config) -> None:
        self._sfreq = config.temporal.sfreq

    def apply(
        self,
        eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        C, T = eeg.shape
        t = np.arange(T) / self._sfreq
        blink_t = float(rng.uniform(0.1, 0.7))
        width = float(rng.uniform(0.05, 0.15))
        eeg_rms = float(np.sqrt(np.mean(eeg ** 2)))
        amplitude = float(rng.uniform(5.0, 15.0)) * (eeg_rms + 1e-10)
        template = (amplitude * np.exp(-0.5 * ((t - blink_t) / width) ** 2)).astype(np.float32)
        channel_weights = rng.uniform(0.5, 1.0, size=(C, 1)).astype(np.float32)
        return (eeg + channel_weights * template[np.newaxis, :]).astype(np.float32)


class MuscularArtifact(ArtifactEngine):
    """High-frequency (30-100 Hz) Gaussian-envelope burst on a random subset of channels."""

    def __init__(self, config) -> None:
        self._sfreq = config.temporal.sfreq

    def apply(
        self,
        eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        C, T = eeg.shape
        t = np.arange(T) / self._sfreq
        n_affected = int(rng.integers(1, min(C, max(2, C // 4)) + 1))
        affected = rng.choice(C, size=n_affected, replace=False)
        burst_t = float(rng.uniform(0.1, 0.7))
        burst_width = float(rng.uniform(0.05, 0.15))
        freq = float(rng.uniform(30.0, 100.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        eeg_rms = float(np.sqrt(np.mean(eeg ** 2)))
        amplitude = float(rng.uniform(2.0, 8.0)) * (eeg_rms + 1e-10)
        envelope = np.exp(-0.5 * ((t - burst_t) / burst_width) ** 2)
        burst = (amplitude * envelope * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
        result = eeg.copy()
        result[affected] += burst[np.newaxis, :]
        return result.astype(np.float32)


class LineNoiseArtifact(ArtifactEngine):
    """50 Hz or 60 Hz sinusoidal line noise added uniformly to all channels."""

    def __init__(self, config) -> None:
        self._sfreq = config.temporal.sfreq

    def apply(
        self,
        eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        C, T = eeg.shape
        t = np.arange(T) / self._sfreq
        freq = float(rng.choice([50.0, 60.0]))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        eeg_rms = float(np.sqrt(np.mean(eeg ** 2)))
        amplitude = float(rng.uniform(0.5, 2.0)) * (eeg_rms + 1e-10)
        line = (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
        return (eeg + line[np.newaxis, :]).astype(np.float32)


class BadChannelDropout(ArtifactEngine):
    """Zero out 1-max(2, C//8) randomly chosen channels."""

    def apply(
        self,
        eeg: np.ndarray,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        C, T = eeg.shape
        n_drop = int(rng.integers(1, max(2, C // 8) + 1))
        channels = rng.choice(C, size=n_drop, replace=False)
        result = eeg.copy()
        result[channels] = 0.0
        return result.astype(np.float32)
