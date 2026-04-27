from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class PriorFamilyWeights(BaseModel):
    local_contiguous: float = 0.30
    network_aware: float = 0.25
    state_dependent: float = 0.20
    broad_random: float = 0.15
    tvb_stub: float = 0.10

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> PriorFamilyWeights:
        total = (
            self.local_contiguous
            + self.network_aware
            + self.state_dependent
            + self.broad_random
            + self.tvb_stub
        )
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Prior family weights must sum to 1.0, got {total:.4f}")
        return self


class NSourcesWeights(BaseModel):
    weights: dict[int, float] = Field(
        default_factory=lambda: {1: 0.20, 2: 0.25, 3: 0.30, 4: 0.25}
    )

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> NSourcesWeights:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"n_sources weights must sum to 1.0, got {total:.4f}")
        return self


class DifficultyWeights(BaseModel):
    easy: float = 0.40
    medium: float = 0.35
    hard: float = 0.25

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> DifficultyWeights:
        total = self.easy + self.medium + self.hard
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Difficulty weights must sum to 1.0, got {total:.4f}")
        return self


class ScenarioPlanConfig(BaseModel):
    prior_family_weights: PriorFamilyWeights = Field(default_factory=PriorFamilyWeights)
    n_sources_weights: NSourcesWeights = Field(default_factory=NSourcesWeights)
    difficulty_weights: DifficultyWeights = Field(default_factory=DifficultyWeights)
    ood_fraction: float = Field(default=0.10, ge=0.0, le=1.0)


class MontageEntry(BaseModel):
    name: str
    n_channels: int
    split_role: Literal["core", "ood"]


class AnatomyBankConfig(BaseModel):
    bank_dir: Path
    anatomy_ids: list[str] = ["fsaverage", "mne_sample", "nyhead"]
    parcellation_scheme: str = "desikan_killiany"


class LeadfieldBankConfig(BaseModel):
    bank_dir: Path
    conductivity_ids: list[str] = ["standard"]


class MontageBankConfig(BaseModel):
    bank_dir: Path
    montages: list[MontageEntry] = Field(default_factory=list)


class ConnectivityBankConfig(BaseModel):
    bank_dir: Path
    schemes: list[str] = ["desikan_killiany"]


class TemporalConfig(BaseModel):
    sfreq: float = 500.0
    window_s: float = 1.0
    signal_families: list[str] = [
        "erp",
        "oscillatory_burst",
        "ar_correlated",
        "spike_interictal",
    ]

    @property
    def n_samples_per_window(self) -> int:
        return int(self.sfreq * self.window_s)


class BackgroundConfig(BaseModel):
    families: list[str] = ["nmm_background", "colored_1f", "ar_background"]
    weights: list[float] = [0.4, 0.4, 0.2]

    @model_validator(mode="after")
    def weights_match_families(self) -> BackgroundConfig:
        if len(self.families) != len(self.weights):
            raise ValueError("Background families and weights must have the same length")
        if abs(sum(self.weights) - 1.0) > 1e-5:
            raise ValueError("Background weights must sum to 1.0")
        return self


class NoiseConfig(BaseModel):
    families: list[str] = ["white_gaussian", "colored_1f", "empirical_resting"]
    weights: list[float] = [0.5, 0.3, 0.2]
    snr_core_range_db: tuple[float, float] = (0.0, 20.0)
    snr_hard_range_db: tuple[float, float] = (-5.0, 30.0)

    @model_validator(mode="after")
    def weights_match_families(self) -> NoiseConfig:
        if len(self.families) != len(self.weights):
            raise ValueError("Noise families and weights must have the same length")
        if abs(sum(self.weights) - 1.0) > 1e-5:
            raise ValueError("Noise weights must sum to 1.0")
        return self


class ArtifactConfig(BaseModel):
    families: list[str] = ["ocular", "muscular", "line_noise", "bad_channel_dropout"]
    artifact_prob: float = Field(default=0.30, ge=0.0, le=1.0)


class ReferenceConfig(BaseModel):
    scheme: Literal["average", "fixed", "none"] = "average"
    fixed_channel: str | None = None


class QCConfig(BaseModel):
    min_valid_channels: int = 10
    max_snir_deviation_db: float = 5.0
    min_inter_source_distance_mm: float = 10.0


class WriterConfig(BaseModel):
    output_dir: Path
    chunk_size: int = Field(default=256, gt=0)


class TVBBackendConfig(BaseModel):
    model: Literal["jansen_rit", "generic_2d_oscillator", "wilson_cowan"] = "jansen_rit"
    connectivity_scheme: str = "desikan_killiany"
    global_coupling: float = Field(default=0.1, ge=0.0)
    noise_sigma: float = Field(default=1e-5, ge=0.0)
    integrator_dt_ms: float = Field(default=1.0, gt=0.0)
    warmup_s: float = Field(default=5.0, ge=0.0)
    reservoir_duration_s: float = Field(default=60.0, gt=0.0)
    reservoir_size: int = Field(default=100, gt=0)
    stimulus_amplitude: float = Field(default=1.0, ge=0.0)
    conduction_speed_mm_per_ms: float = Field(default=3.0, gt=0.0)


class GenerationConfig(BaseModel):
    anatomy_bank: AnatomyBankConfig
    leadfield_bank: LeadfieldBankConfig
    montage_bank: MontageBankConfig
    connectivity_bank: ConnectivityBankConfig = Field(
        default_factory=lambda: ConnectivityBankConfig(bank_dir=Path("banks/connectivity"))
    )
    scenario_plan: ScenarioPlanConfig = Field(default_factory=ScenarioPlanConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    writer: WriterConfig
    backend: Literal["sereega", "tvb"] = "sereega"
    tvb: TVBBackendConfig = Field(default_factory=TVBBackendConfig)
    n_samples: int = Field(default=100_000, gt=0)
    n_workers: int = Field(default=4, gt=0)
    global_seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> GenerationConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
