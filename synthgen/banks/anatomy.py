from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from synthgen.config import AnatomyBankConfig
from synthgen.sample import SourceSpace


class AnatomyBank:
    def __init__(self, config: AnatomyBankConfig) -> None:
        self._bank_dir = Path(config.bank_dir)
        self._default_scheme = config.parcellation_scheme
        self._cache: dict[tuple[str, str], SourceSpace] = {}

    def load(self, anatomy_id: str, scheme: str | None = None) -> SourceSpace:
        scheme = scheme or self._default_scheme
        key = (anatomy_id, scheme)
        if key in self._cache:
            return self._cache[key]

        ss_path = self._bank_dir / anatomy_id / "source_space.npz"
        if not ss_path.exists():
            raise FileNotFoundError(f"Source space not found: {ss_path}")
        data = np.load(ss_path, allow_pickle=False)
        adjacency = sp.csr_matrix(
            (
                data["adjacency_data"],
                data["adjacency_indices"],
                data["adjacency_indptr"],
            ),
            shape=tuple(data["adjacency_shape"]),
        )

        parc_path = self._bank_dir / anatomy_id / "parcellations" / f"{scheme}.npz"
        if not parc_path.exists():
            raise FileNotFoundError(f"Parcellation not found: {parc_path}")
        pdata = np.load(parc_path, allow_pickle=False)
        region_labels = [str(s) for s in pdata["region_labels"].tolist()]
        stored_scheme = str(pdata["scheme"])
        if stored_scheme != scheme:
            raise ValueError(
                f"Parcellation file {parc_path} declares scheme={stored_scheme!r}, expected {scheme!r}"
            )

        ss = SourceSpace(
            vertex_coords=data["vertex_coords"],
            adjacency=adjacency,
            parcellation=pdata["parcellation"],
            hemisphere=data["hemisphere"],
            parcellation_scheme=scheme,
            region_labels=region_labels,
        )
        self._cache[key] = ss
        return ss

    def available(self) -> list[str]:
        if not self._bank_dir.exists():
            return []
        return [d.name for d in self._bank_dir.iterdir() if d.is_dir()]

    def available_schemes(self, anatomy_id: str) -> list[str]:
        parc_dir = self._bank_dir / anatomy_id / "parcellations"
        if not parc_dir.exists():
            return []
        return sorted(p.stem for p in parc_dir.glob("*.npz"))
