from __future__ import annotations

from pathlib import Path

import numpy as np

from synthgen.config import LeadfieldBankConfig


class LeadfieldBank:
    def __init__(self, config: LeadfieldBankConfig) -> None:
        self._bank_dir = Path(config.bank_dir)
        self._cache: dict[str, np.ndarray] = {}

    def load(self, leadfield_id: str) -> np.ndarray:
        if leadfield_id in self._cache:
            return self._cache[leadfield_id]
        parts = leadfield_id.split("__", maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"leadfield_id must be 'anatomy__montage__conductivity', got: {leadfield_id!r}")
        anatomy_id, montage_id, conductivity_id = parts
        path = self._bank_dir / anatomy_id / montage_id / f"{conductivity_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Leadfield not found: {path}")
        G = np.load(path, allow_pickle=False)["G"]
        self._cache[leadfield_id] = G
        return G
