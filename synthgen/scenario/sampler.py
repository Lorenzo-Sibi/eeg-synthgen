from __future__ import annotations

import numpy as np

from synthgen.config import GenerationConfig
from synthgen.sample import Scenario


def _weighted_choice(keys: list, weights: list[float], rng: np.random.Generator) -> str:
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    idx = int(rng.choice(len(keys), p=w))
    return str(keys[idx])


class ScenarioSampler:
    """Draws a fully-specified Scenario from config-weighted distributions."""

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config
        plan = config.scenario_plan
        self._prior_families = list(type(plan.prior_family_weights).model_fields.keys())
        self._prior_weights = [
            getattr(plan.prior_family_weights, f) for f in self._prior_families
        ]
        ns = plan.n_sources_weights.weights
        self._n_sources_keys = [int(k) for k in ns.keys()]
        self._n_sources_weights = list(ns.values())
        dw = plan.difficulty_weights
        self._difficulties = ["easy", "medium", "hard"]
        self._difficulty_weights = [dw.easy, dw.medium, dw.hard]
        self._ood_fraction = plan.ood_fraction
        self._core_montages = [m for m in config.montages.montages if m.split_role == "core"]
        if not self._core_montages:
            raise ValueError(
                "No core montages found in config; at least one montage must have split_role='core'"
            )
        self._ood_montages = [m for m in config.montages.montages if m.split_role == "ood"]
        self._conductivity_ids = config.leadfield_bank.conductivity_ids
        self._anatomy_ids = config.anatomy_bank.anatomy_ids
        self._signal_families = config.temporal.signal_families
        self._signal_family_weights = config.temporal.signal_family_weights

    def sample(self, rng: np.random.Generator) -> Scenario:
        seed = int(rng.integers(0, 2 ** 63))

        if self._ood_montages and float(rng.uniform()) < self._ood_fraction:
            montage = self._ood_montages[int(rng.integers(0, len(self._ood_montages)))]
            split = "ood"
        else:
            montage = self._core_montages[int(rng.integers(0, len(self._core_montages)))]
            split = "train"

        anatomy_id = str(self._anatomy_ids[int(rng.integers(0, len(self._anatomy_ids)))])
        conductivity_id = str(self._conductivity_ids[int(rng.integers(0, len(self._conductivity_ids)))])
        leadfield_id = f"{anatomy_id}__{montage.name}__{conductivity_id}"

        prior_family = _weighted_choice(self._prior_families, self._prior_weights, rng)
        n_sources = int(_weighted_choice(
            [str(k) for k in self._n_sources_keys],
            self._n_sources_weights,
            rng,
        ))
        difficulty = _weighted_choice(self._difficulties, self._difficulty_weights, rng)
        if self._signal_family_weights is None:
            signal_family = str(
                self._signal_families[int(rng.integers(0, len(self._signal_families)))]
            )
        else:
            signal_family = _weighted_choice(
                self._signal_families,
                self._signal_family_weights,
                rng,
            )

        scenario_id = f"s{seed:020d}"

        return Scenario(
            scenario_id=scenario_id,
            seed=seed,
            anatomy_id=anatomy_id,
            leadfield_id=leadfield_id,
            montage_id=montage.name,
            reference_scheme=self._config.reference.scheme,
            conductivity_id=conductivity_id,
            prior_family=prior_family,
            n_sources=n_sources,
            signal_family=signal_family,
            difficulty=difficulty,
            split=split,
        )
