from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthgen.config import MontageBankConfig


@dataclass
class MontageData:
    coords: np.ndarray    # (C, 3) float32, mm, head coordinate frame
    ch_names: list[str]


class MontageBank:
    def __init__(self, config: MontageBankConfig) -> None:
        self._bank_dir = Path(config.bank_dir)
        self._cache: dict[str, MontageData] = {}

    def load(self, montage_id: str) -> MontageData:
        if montage_id in self._cache:
            return self._cache[montage_id]
        path = self._bank_dir / f"{montage_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Montage not found: {path}")
        data = np.load(path, allow_pickle=False)
        m = MontageData(
            coords=data["coords"].astype(np.float32),
            ch_names=[str(s) for s in data["ch_names"]],
        )
        self._cache[montage_id] = m
        return m
