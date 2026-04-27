from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse


@dataclass
class SourceSpace:
    vertex_coords: np.ndarray
    adjacency: scipy.sparse.spmatrix
    parcellation: np.ndarray
    hemisphere: np.ndarray
    parcellation_scheme: str = "hemisphere" # e.g. "desikan"
    region_labels: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    scenario_id: str
    seed: int
    anatomy_id: str
    leadfield_id: str
    montage_id: str
    reference_scheme: str
    conductivity_id: str
    prior_family: str
    n_sources: int
    signal_family: str
    difficulty: str
    split: str

    patch_extents_cm2: list[float] = field(default_factory=list)
    seed_vertex_indices: list[int] = field(default_factory=list)
    seed_patch_vertex_indices: list[list[int]] = field(default_factory=list)
    inter_source_distances_mm: list[float] = field(default_factory=list)
    source_correlation: float = 0.0
    temporal_onsets_s: list[float] = field(default_factory=list)
    dominant_frequencies_hz: list[float] = field(default_factory=list)

    snir_db: float = 0.0
    snr_sensor_db: float = 0.0
    artifact_flags: list[str] = field(default_factory=list)


@dataclass
class EEGSample:
    eeg: np.ndarray
    source_activity: np.ndarray
    source_support: np.ndarray
    electrode_coords: np.ndarray
    source_coords: np.ndarray
    params: Scenario
    snir_measured_db: float
    snr_sensor_measured_db: float
    active_area_cm2: float
    config_hash: str