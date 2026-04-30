from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthgen.config import LeadfieldBankConfig


@dataclass
class LeadfieldData:
    G: np.ndarray                  # (C, N) float32
    ch_names: list[str]
    electrode_coords: np.ndarray   # (C, 3) float32, mm, HEAD frame


class LeadfieldBank:
    def __init__(self, config: LeadfieldBankConfig) -> None:
        self._bank_dir = Path(config.bank_dir)
        self._cache: dict[str, LeadfieldData] = {}

    def load(self, leadfield_id: str) -> LeadfieldData:
        if leadfield_id in self._cache:
            return self._cache[leadfield_id]
        parts = leadfield_id.split("__", maxsplit=2)
        if len(parts) != 3:
            raise ValueError(
                f"leadfield_id must be 'anatomy__montage__conductivity', got: {leadfield_id!r}"
            )
        anatomy_id, montage_id, conductivity_id = parts
        path = self._bank_dir / anatomy_id / montage_id / f"{conductivity_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Leadfield not found: {path}")
        data = np.load(path, allow_pickle=False)
        lf = LeadfieldData(
            G=data["G"].astype(np.float32),
            ch_names=[str(s) for s in data["ch_names"]],
            electrode_coords=data["electrode_coords"].astype(np.float32),
        )
        self._cache[leadfield_id] = lf
        return lf
