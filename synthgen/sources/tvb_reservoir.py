from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StateSample:
    state: np.ndarray


class StateReservoir:
    """Pre-warmed TVB state vectors sampled at random times of a long sim.

    A single long run of duration ``warmup_s + reservoir_duration_s`` is executed
    at build time. Post-warmup samples are decimated uniformly to ``n_states``.
    ``sample(rng)`` draws one snapshot with replacement.
    """

    def __init__(
        self,
        simulator,
        warmup_s: float,
        reservoir_duration_s: float,
        n_states: int,
    ) -> None:
        duration_ms = (warmup_s + reservoir_duration_s) * 1000.0
        simulator.configure()
        simulator.simulation_length = duration_ms
        (raw_time, raw_state), = simulator.run()
        raw_time = np.asarray(raw_time)
        raw_state = np.asarray(raw_state)

        mask = raw_time >= warmup_s * 1000.0
        keep_state = raw_state[mask]
        if keep_state.shape[0] < n_states:
            raise ValueError(
                f"Reservoir warmup window produced only {keep_state.shape[0]} samples, "
                f"need >= {n_states}. Increase reservoir_duration_s."
            )
        idx = np.linspace(0, keep_state.shape[0] - 1, n_states).astype(np.int64)
        self._states = keep_state[idx].copy()

    def __len__(self) -> int:
        return self._states.shape[0]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        i = int(rng.integers(0, self._states.shape[0]))
        return self._states[i].copy()
