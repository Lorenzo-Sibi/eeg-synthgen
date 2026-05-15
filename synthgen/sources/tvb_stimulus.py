from __future__ import annotations

import numpy as np

from synthgen.sample import Scenario


def seed_parcel_ids(scenario: Scenario, parcellation: np.ndarray) -> np.ndarray:
    """Return unique parcel ids touched by the scenario's seed patches.

    Falls back to ``seed_vertex_indices`` when ``seed_patch_vertex_indices``
    is empty (e.g. legacy scenarios or tests that build Scenarios by hand).

    Note: the DeepSIF-style TVB backend no longer injects an external stimulus
    pattern. Seed parcels are identified here only so the simulator can
    elevate the local Jansen-Rit excitability ``A`` parameter on them.
    """
    patches = scenario.seed_patch_vertex_indices
    if patches:
        flat: list[int] = []
        for p in patches:
            flat.extend(int(v) for v in p)
        if not flat:
            return np.array([], dtype=np.int64)
        ids = np.unique(parcellation[np.asarray(flat, dtype=np.int64)])
        return ids.astype(np.int64)
    if not scenario.seed_vertex_indices:
        return np.array([], dtype=np.int64)
    ids = np.unique(parcellation[np.asarray(scenario.seed_vertex_indices, dtype=np.int64)])
    return ids.astype(np.int64)
