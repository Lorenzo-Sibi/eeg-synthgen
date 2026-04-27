from __future__ import annotations

import numpy as np

from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.priors.base import ActivationPrior
from synthgen.sources.priors._helpers import (
    _compute_patches,
    _pairwise_distances_mm,
    _sample_frequencies,
)


class NetworkAwarePrior(ActivationPrior):
    def sample(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        rng: np.random.Generator,
    ) -> Scenario:
        N = len(source_space.vertex_coords)
        n = scenario.n_sources

        unique_parcels = np.unique(source_space.parcellation)
        parcel = int(rng.choice(unique_parcels))
        parcel_verts = np.where(source_space.parcellation == parcel)[0]
        if len(parcel_verts) < n:
            parcel_verts = np.arange(N)

        seeds = [int(rng.choice(parcel_verts)) for _ in range(n)]
        scenario.seed_vertex_indices = seeds
        scenario.patch_extents_cm2 = [float(rng.uniform(0.5, 3.0)) for _ in range(n)]
        scenario.seed_patch_vertex_indices = _compute_patches(
            source_space, seeds, scenario.patch_extents_cm2
        )
        scenario.inter_source_distances_mm = _pairwise_distances_mm(source_space.vertex_coords, seeds)
        scenario.source_correlation = float(rng.uniform(0.3, 0.8))
        scenario.temporal_onsets_s = [float(rng.uniform(0.0, 0.3)) for _ in range(n)]
        scenario.dominant_frequencies_hz = _sample_frequencies(scenario.signal_family, n, rng)
        return scenario
