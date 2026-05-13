#!/usr/bin/env python3
"""Validate a generated synthetic EEG dataset.

Usage:
    python scripts/validate_dataset.py data/generated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def validate(dataset_dir: Path) -> None:
    zarr_path = dataset_dir / "data.zarr"
    jsonl_path = dataset_dir / "metadata.jsonl"

    assert zarr_path.exists(), f"Zarr store not found: {zarr_path}"
    assert jsonl_path.exists(), f"Metadata file not found: {jsonl_path}"

    store = zarr.open(str(zarr_path), mode="r")
    groups = list(store.group_keys())
    assert len(groups) > 0, "No data groups found in zarr store"

    n_total = 0
    for key in groups:
        grp = store[key]
        n = grp["eeg"].shape[0]
        n_total += n
        print(f"  Group '{key}': {n} samples, eeg shape {grp['eeg'].shape}")

    manifest_n = store.attrs.get("n_samples", None)
    if manifest_n is not None:
        assert manifest_n == n_total, (
            f"Manifest says {manifest_n} samples but found {n_total}"
        )

    with open(jsonl_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == n_total, (
        f"metadata.jsonl has {len(lines)} lines but zarr has {n_total} samples"
    )

    # Quick sanity: parse first line
    if lines:
        first = json.loads(lines[0])
        assert "scenario_id" in first, "metadata.jsonl missing scenario_id field"
        assert "prior_family" in first, "metadata.jsonl missing prior_family field"

    # SIR/SNR/SINR stats from first group (targets recorded at sampling time)
    key0 = groups[0]
    sir = np.array(store[key0]["sir_db"])
    snr = np.array(store[key0]["snr_db"])
    sinr = np.array(store[key0]["sinr_db"])
    print(f"\nSIR  (group '{key0}'): {sir.mean():.1f} ± {sir.std():.1f} dB")
    print(f"SNR  (group '{key0}'): {snr.mean():.1f} ± {snr.std():.1f} dB")
    print(f"SINR (group '{key0}'): {sinr.mean():.1f} ± {sinr.std():.1f} dB")

    print(f"\nTotal samples: {n_total}")
    print("Validation passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generated EEG dataset.")
    parser.add_argument("dataset", type=Path, help="Path to dataset directory")
    args = parser.parse_args()
    validate(args.dataset)


if __name__ == "__main__":
    main()
