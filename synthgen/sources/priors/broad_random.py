from __future__ import annotations

import numpy as np

from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.priors.base import ActivationPrior
from synthgen.sources.priors._helpers import (
    _compute_patches,
    _pairwise_distances_mm,
    _sample_frequencies,
)


class BroadRandomPrior(ActivationPrior):
    def sample(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        rng: np.random.Generator,
    ) -> Scenario:
        N = len(source_space.vertex_coords)
        n = scenario.n_sources
        seeds = [int(rng.integers(0, N)) for _ in range(n)]
        scenario.seed_vertex_indices = seeds
        scenario.patch_extents_cm2 = [float(rng.uniform(0.5, 10.0)) for _ in range(n)]
        scenario.seed_patch_vertex_indices = _compute_patches(
            source_space, seeds, scenario.patch_extents_cm2
        )
        scenario.inter_source_distances_mm = _pairwise_distances_mm(source_space.vertex_coords, seeds)
        scenario.source_correlation = float(rng.uniform(0.0, 1.0))
        scenario.temporal_onsets_s = [float(rng.uniform(0.0, 0.5)) for _ in range(n)]
        scenario.dominant_frequencies_hz = _sample_frequencies(scenario.signal_family, n, rng)
        return scenario
