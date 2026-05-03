from __future__ import annotations

import numpy as np

from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.priors.base import ActivationPrior
from synthgen.sources.priors._helpers import (
    _compute_patches,
    _pairwise_distances_mm,
    _sample_frequencies,
)

_EXTENT_RANGES: dict[str, tuple[float, float]] = {
    "easy": (2.0, 8.0),
    "medium": (1.0, 4.0),
    "hard": (0.5, 2.0),
}


class LocalContiguousPrior(ActivationPrior):
    """Activations are spatially contiguous patches around random seed vertices."""
    
    def sample(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        rng: np.random.Generator,
    ) -> Scenario:
        N = len(source_space.vertex_coords)
        n = scenario.n_sources
        lo, hi = _EXTENT_RANGES.get(scenario.difficulty, (1.0, 4.0))
        seeds = rng.integers(0, N, size=n).tolist() # here we sample the seed vertices
        scenario.seed_vertex_indices = seeds
        scenario.patch_extents_cm2 = rng.uniform(lo, hi, size=n).tolist()
        scenario.seed_patch_vertex_indices = _compute_patches(
            source_space, seeds, scenario.patch_extents_cm2
        )
        scenario.inter_source_distances_mm = _pairwise_distances_mm(source_space.vertex_coords, seeds)
        scenario.source_correlation = float(rng.uniform(0.0, 0.3))
        scenario.temporal_onsets_s = rng.uniform(0.0, 0.4, size=n).tolist()
        scenario.dominant_frequencies_hz = _sample_frequencies(scenario.signal_family, n, rng)
        return scenario
