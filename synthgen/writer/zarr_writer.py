from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import zarr

from synthgen.config import WriterConfig
from synthgen.sample import EEGSample
from synthgen.writer.base import DatasetWriter


class ZarrWriter(DatasetWriter):
    """Buffered zarr v3 writer.

    Groups samples by ``anatomy_id__montage_id`` so arrays have fixed C and V
    dimensions. Scenario metadata is written to a companion ``metadata.jsonl``.
    """

    def __init__(self, config: WriterConfig) -> None:
        self._output_dir = Path(config.output_dir)
        self._chunk_size = config.chunk_size
        self._buf: dict[str, list[EEGSample]] = {}
        self._n_written: dict[str, int] = {}
        self._store: zarr.Group | None = None
        self._jsonl_path = self._output_dir / "metadata.jsonl"
        self._jsonl_file = None

    @staticmethod
    def _group_key(sample: EEGSample) -> str:
        sc = sample.params
        return f"{sc.anatomy_id}__{sc.montage_id}"

    def write(self, sample: EEGSample) -> None:
        key = self._group_key(sample)
        self._buf.setdefault(key, []).append(sample)
        if len(self._buf[key]) >= self._chunk_size:
            self._flush_group(key)

    def _init_store(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._store = zarr.open(str(self._output_dir / "data.zarr"), mode="w")
        self._jsonl_file = open(self._jsonl_path, "w")

    def _init_group(self, key: str, sample: EEGSample) -> None:
        C, T = sample.eeg.shape
        V = sample.source_support.shape[0]
        cs = self._chunk_size
        grp = self._store.create_group(key)
        grp.create_array("eeg", shape=(0, C, T), dtype="float32", chunks=(cs, C, T))
        grp.create_array("source_support", shape=(0, V), dtype="bool", chunks=(cs, V))
        grp.create_array("snir_db", shape=(0,), dtype="float32", chunks=(cs,))
        grp.create_array("snr_sensor_db", shape=(0,), dtype="float32", chunks=(cs,))
        grp.create_array("active_area_cm2", shape=(0,), dtype="float32", chunks=(cs,))
        self._n_written[key] = 0

    def _flush_group(self, key: str) -> None:
        buf = self._buf.pop(key, [])
        if not buf:
            return
        if self._store is None:
            self._init_store()
        if key not in self._n_written:
            self._init_group(key, buf[0])

        n = len(buf)
        n0 = self._n_written[key]
        grp = self._store[key]

        eeg_batch = np.stack([s.eeg for s in buf])
        sup_batch = np.stack([s.source_support for s in buf])

        for arr_name, batch in [("eeg", eeg_batch), ("source_support", sup_batch)]:
            arr = grp[arr_name]
            new_shape = (n0 + n, *arr.shape[1:])
            arr.resize(new_shape)
            arr[n0:n0 + n] = batch

        for arr_name, vals in [
            ("snir_db",         [s.params.snir_db for s in buf]),
            ("snr_sensor_db",   [s.params.snr_sensor_db for s in buf]),
            ("active_area_cm2", [s.active_area_cm2 for s in buf]),
        ]:
            arr = grp[arr_name]
            arr.resize((n0 + n,))
            arr[n0:n0 + n] = np.array(vals, dtype=np.float32)

        self._n_written[key] += n

        for s in buf:
            self._jsonl_file.write(json.dumps(asdict(s.params)) + "\n")

    def finalize(self) -> None:
        for key in list(self._buf.keys()):
            self._flush_group(key)
        n_total = sum(self._n_written.values())
        if self._store is not None:
            self._store.attrs["n_samples"] = n_total
            self._store.attrs["schema_version"] = "1"
            self._store = None
        if self._jsonl_file is not None:
            self._jsonl_file.flush()
            self._jsonl_file.close()
            self._jsonl_file = None

    def __enter__(self) -> "ZarrWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finalize()
