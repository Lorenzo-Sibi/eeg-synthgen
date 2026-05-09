from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd
import pytest

from synthgen.analysis.plotting import boxplot_per_montage, boxplot_per_snr_bin


def _synthetic_records(n_samples: int = 20):
    rng = np.random.RandomState(0)
    records = []
    methods = ["MNE", "sLORETA", "eLORETA"]
    montages = ["mtg_64", "mtg_32"]
    for _ in range(n_samples):
        for method in methods:
            for montage in montages:
                records.append({
                    "method": method,
                    "montage_id": montage,
                    "snr_bin_db": int(rng.choice([0, 5, 10])),
                    "le_mm": float(abs(rng.randn() * 5 + 10)),
                    "te_ms": float(abs(rng.randn() * 10 + 30)),
                    "nmse": float(abs(rng.randn() * 0.05 + 0.1)),
                    "psnr_db": float(rng.randn() * 3 + 30),
                    "auc": float(np.clip(rng.rand() * 0.3 + 0.7, 0, 1)),
                })
    return records


def test_boxplot_per_montage_writes_pdf_and_png(tmp_path: Path):
    records = _synthetic_records()
    metrics = ["le_mm", "te_ms", "nmse", "psnr_db", "auc"]
    boxplot_per_montage(records, metrics, tmp_path)
    assert (tmp_path / "fig_metrics_per_montage.pdf").exists()
    assert (tmp_path / "fig_metrics_per_montage.png").exists()


def test_boxplot_per_snr_writes_files(tmp_path: Path):
    records = _synthetic_records()
    metrics = ["le_mm", "te_ms", "nmse", "psnr_db", "auc"]
    boxplot_per_snr_bin(records, metrics, tmp_path)
    assert (tmp_path / "fig_metrics_per_snr.pdf").exists()
    assert (tmp_path / "fig_metrics_per_snr.png").exists()
