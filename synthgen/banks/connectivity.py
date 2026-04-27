from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthgen.config import ConnectivityBankConfig


@dataclass
class Connectivity:
    weights: np.ndarray
    tract_lengths: np.ndarray
    region_centers: np.ndarray
    region_labels: list[str]
    scheme: str


class ConnectivityBank:
    def __init__(self, config: ConnectivityBankConfig) -> None:
        self._bank_dir = Path(config.bank_dir)
        self._cache: dict[str, Connectivity] = {}

    def load(self, scheme: str) -> Connectivity:
        if scheme in self._cache:
            return self._cache[scheme]
        path = self._bank_dir / f"{scheme}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Connectivity not found: {path}")
        d = np.load(path, allow_pickle=False)
        labels = [str(s) for s in d["region_labels"].tolist()]
        conn = Connectivity(
            weights=d["weights"].astype(np.float32),
            tract_lengths=d["tract_lengths"].astype(np.float32),
            region_centers=d["region_centers"].astype(np.float32),
            region_labels=labels,
            scheme=str(d["scheme"]),
        )
        self._cache[scheme] = conn
        return conn