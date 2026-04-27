from __future__ import annotations

import numpy as np

from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.priors.base import ActivationPrior
from synthgen.sources.priors._helpers import _compute_patches, _pairwise_distances_mm

_BRAIN_STATES: dict[str, dict] = {
    "alpha": {
        "freq_range": (8.0, 13.0),
        "signal_family": "oscillatory_burst",
    },
    "mu_beta": {
        "freq_range": (12.0, 30.0),
        "signal_family": "oscillatory_burst",
    },
    "dmn": {
        "freq_range": (2.0, 8.0),
        "signal_family": "ar_correlated",
    },
    "epileptiform": {
        "freq_range": (1.0, 4.0),
        "signal_family": "spike_interictal",
    },
}


class StateDependentPrior(ActivationPrior):
    def sample(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        rng: np.random.Generator,
    ) -> Scenario:
        N = len(source_space.vertex_coords)
        n = scenario.n_sources

        state_name = str(rng.choice(list(_BRAIN_STATES.keys())))
        state = _BRAIN_STATES[state_name]
        scenario.signal_family = state["signal_family"]

        seeds = [int(rng.integers(0, N)) for _ in range(n)]
        freq_lo, freq_hi = state["freq_range"]

        scenario.seed_vertex_indices = seeds
        scenario.patch_extents_cm2 = [float(rng.uniform(1.0, 5.0)) for _ in range(n)]
        scenario.seed_patch_vertex_indices = _compute_patches(
            source_space, seeds, scenario.patch_extents_cm2
        )
        scenario.inter_source_distances_mm = _pairwise_distances_mm(source_space.vertex_coords, seeds)
        scenario.source_correlation = float(rng.uniform(0.2, 0.7))
        scenario.temporal_onsets_s = [float(rng.uniform(0.0, 0.3)) for _ in range(n)]
        scenario.dominant_frequencies_hz = [float(rng.uniform(freq_lo, freq_hi)) for _ in range(n)]
        return scenario
