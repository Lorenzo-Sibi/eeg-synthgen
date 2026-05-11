from __future__ import annotations

import hashlib
import json

import numpy as np

from synthgen.acquisition.artifacts import (
    ArtifactEngine,
    BadChannelDropout,
    LineNoiseArtifact,
    MuscularArtifact,
    OcularArtifact,
)
from synthgen.acquisition.noise import (
    Colored1fNoise,
    EmpiricalRestingNoise,
    SensorNoiseEngine,
    WhiteGaussianNoise,
)
from synthgen.acquisition.qc import QCResult, check_sample
from synthgen.acquisition.reference import AverageReference, FixedReference, NoReference
from synthgen.config import GenerationConfig, ReferenceConfig
from synthgen.forward.projector import LinearProjector
from synthgen.sample import EEGSample, Scenario, SourceSpace


def _make_reference_op(ref_config: ReferenceConfig, ch_names: list[str]):
    if ref_config.scheme == "average":
        return AverageReference()
    if ref_config.scheme == "fixed":
        if ref_config.fixed_channel is not None and ref_config.fixed_channel in ch_names:
            return FixedReference(ch_names.index(ref_config.fixed_channel))
        return AverageReference()
    return NoReference()


def _hash_config(config: GenerationConfig) -> str:
    return hashlib.sha256(
        json.dumps(config.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()[:8]


class AcquisitionPipeline:
    """Orchestrates the full acquisition chain: project → mix → noise → artifact → reference."""

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config
        self._projector = LinearProjector()
        self._noise_registry: dict[str, SensorNoiseEngine] = {
            "white_gaussian": WhiteGaussianNoise(),
            "colored_1f": Colored1fNoise(config),
            "empirical_resting": EmpiricalRestingNoise(),
        }
        self._artifact_registry: dict[str, ArtifactEngine] = {
            "ocular": OcularArtifact(config),
            "muscular": MuscularArtifact(config),
            "line_noise": LineNoiseArtifact(config),
            "bad_channel_dropout": BadChannelDropout(),
        }
        self._config_hash = _hash_config(config)

    def run(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        source_activity: np.ndarray,
        background_activity: np.ndarray,
        leadfield: np.ndarray,
        electrode_coords: np.ndarray,
        ch_names: list[str],
        rng: np.random.Generator,
    ) -> tuple[EEGSample, QCResult]:
        """
        Parameters
        ----------
        source_activity:     (N, T) float32 — source-space signal from backend
        background_activity: (N, T) float32 — source-space background from backend
        leadfield:           (C, N) float32 — G matrix from LeadfieldBank
        electrode_coords:    (C, 3) float32 — sensor positions in mm
        ch_names:            list of C channel names

        Note: `scenario` is mutated in-place (snir_db, snr_sensor_db, artifact_flags).
        Pass a distinct Scenario instance per call when running in parallel.
        """
        noise_cfg = self._config.noise
        artifact_cfg = self._config.artifacts

        # 1. Project to sensor space
        signal_eeg = self._projector.project(source_activity, leadfield).astype(np.float32)
        bg_eeg = self._projector.project(background_activity, leadfield).astype(np.float32)

        # 2. Scale background to achieve target SNIR (discrete grid, DeepSIF protocol)
        target_snir_db = float(rng.choice(noise_cfg.snir_levels_db))    # SNIR = S / (b * S_bg)
        signal_rms = float(np.sqrt(np.mean(signal_eeg ** 2)))
        bg_rms = float(np.sqrt(np.mean(bg_eeg ** 2)))
        if signal_rms > 1e-10 and bg_rms > 1e-10:
            bg_scale = signal_rms / (bg_rms * 10.0 ** (target_snir_db / 20.0))
        else:
            # Signal or background silent: SNIR is undefined; emit zero background.
            bg_scale = 0.0
        measured_snir = target_snir_db

        clean_eeg = (signal_eeg + bg_scale * bg_eeg).astype(np.float32) # S + b * S_bg, where b is chosen to meet target SNIR
        scenario.snir_db = measured_snir

        # 3. Sample target sensor SNR (discrete grid) and add noise
        target_snr_db = float(rng.choice(noise_cfg.snr_sensor_levels_db))
        scenario.snr_sensor_db = target_snr_db

        noise_families = noise_cfg.families
        noise_weights = np.array(noise_cfg.weights, dtype=np.float64)
        noise_idx = int(rng.choice(len(noise_families), p=noise_weights / noise_weights.sum()))
        noise_family = noise_families[noise_idx]
        noisy_eeg = self._noise_registry[noise_family].apply(signal_eeg, scenario, rng)

        # 4. Optionally inject an artifact (uniform family choice — ArtifactConfig has no weights)
        if float(rng.uniform(0.0, 1.0)) < artifact_cfg.artifact_prob:
            artifact_idx = int(rng.integers(0, len(artifact_cfg.families)))
            artifact_family = artifact_cfg.families[artifact_idx]
            noisy_eeg = self._artifact_registry[artifact_family].apply(noisy_eeg, scenario, rng)
            scenario.artifact_flags.append(artifact_family)

        # 5. Apply reference
        ref_op = _make_reference_op(self._config.reference, ch_names)
        final_eeg = ref_op.apply(noisy_eeg).astype(np.float32)

        # 6. Measure sensor-level SNR
        signal_power = float(np.mean(signal_eeg ** 2))
        sensor_noise = noisy_eeg - clean_eeg
        sensor_noise_power = float(np.mean(sensor_noise ** 2))
        measured_snr_sensor = float(10.0 * np.log10((signal_power + 1e-20) / (sensor_noise_power + 1e-20)))
        """ 
        total_noise = final_eeg - signal_eeg    # total_noise = (S + b*S_bg + noise + <artifact> ) - S = b*S_bg + noise + <artifact>
        signal_power = float(np.mean(signal_eeg ** 2))
        noise_power = float(np.mean(total_noise ** 2))
        measured_snr_sensor = float(
            10.0 * np.log10((signal_power + 1e-20) / (noise_power + 1e-20))
        )
        """

        # 7. Build source support mask and active area
        N = len(source_space.vertex_coords)
        source_support = np.zeros(N, dtype=bool)
        patches = scenario.seed_patch_vertex_indices
        if patches:
            for patch in patches:
                source_support[np.asarray(patch, dtype=np.int64)] = True
        else:
            for idx in scenario.seed_vertex_indices:
                source_support[idx] = True
        active_area_cm2 = float(sum(scenario.patch_extents_cm2)) if scenario.patch_extents_cm2 else 0.0

        sample = EEGSample(
            eeg=final_eeg,
            source_activity=source_activity,
            source_support=source_support,
            electrode_coords=electrode_coords,
            source_coords=source_space.vertex_coords,
            params=scenario,
            snir_measured_db=measured_snir,
            snr_sensor_measured_db=measured_snr_sensor,
            active_area_cm2=active_area_cm2,
            config_hash=self._config_hash,
        )

        qc_result = check_sample(sample, self._config.qc)
        return sample, qc_result
