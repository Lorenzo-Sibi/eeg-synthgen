#!/usr/bin/env python3
"""Fit noise parameters from PhysioNet EEGMMIDB resting-state EEG.

Uses the PhysioNet EEG Motor Movement/Imagery Database (Schalk et al. 2004),
the most heavily used public EEG dataset in the literature. Runs 1 and 2
are the eyes-open / eyes-closed baseline (resting) recordings. Downloaded
via mne.datasets.eegbci (no external dependency).

Usage:
    python scripts/calibrate_noise.py --config config/default.yaml \
        --n-subjects 30 --out banks/noise/physionet_v1.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from synthgen.config import GenerationConfig

PHYSIONET_CACHE_DIR = Path("banks/noise/_cache/physionet")
PHYSIONET_RUNS = (1, 2)  # 1 = baseline EO, 2 = baseline EC


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--n-subjects", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("banks/noise/physionet_v1.npz"))
    p.add_argument("--cache-dir", type=Path, default=PHYSIONET_CACHE_DIR)
    p.add_argument("--skip-download", action="store_true",
                   help="Use existing cache, do not re-download")
    return p.parse_args()


def download_physionet(cache_dir: Path, n_subjects: int) -> list[Path]:
    """Download n_subjects resting-state baseline runs from PhysioNet EEGMMIDB.

    Returns the list of per-subject EDF paths (one per subject, run 1 or 2).
    """
    from mne.datasets import eegbci

    subject_ids = list(range(1, n_subjects + 1))
    raw_paths: list[Path] = []
    for sub in subject_ids:
        try:
            paths = eegbci.load_data(
                subjects=[sub], runs=list(PHYSIONET_RUNS),
                path=str(cache_dir), update_path=False, verbose="ERROR",
            )
        except Exception as exc:
            print(f"  skip sub-{sub:03d}: {exc}")
            continue
        for p in paths:
            raw_paths.append(Path(p))
            print(f"  ok sub-{sub:03d}: {Path(p).name}")
    if len(raw_paths) < n_subjects:
        print(f"WARNING: only {len(raw_paths)} EDF files across {n_subjects} subjects")
    return raw_paths


def preprocess_recordings(
    raw_paths: list[Path],
    sfreq: float,
    window_s: float,
    band_hz: tuple[float, float] = (1.0, 40.0),
) -> tuple[list[np.ndarray], list[str]]:
    """Read, standardise channels, filter, resample, epoch each EDF recording.

    PhysioNet EEGMMIDB ships at 160 Hz with BCI2000 channel labels (Fc5., ...).
    `eegbci.standardize` rewrites them to the 10-20 system and applies the
    standard 1005 montage. Band-pass cuts at min(40, Nyquist/2).

    Returns:
        epochs_list: one (C, T) array per epoch, T = int(sfreq * window_s)
        ch_names:    channel names (uniform across all returned epochs)
    """
    import mne
    from mne.datasets import eegbci

    T = int(sfreq * window_s)
    nyq = sfreq / 2.0
    hi = min(band_hz[1], 0.95 * nyq)
    epochs_list: list[np.ndarray] = []
    ch_names_ref: list[str] | None = None
    for path in raw_paths:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        eegbci.standardize(raw)  # in-place channel renaming
        raw.set_montage("standard_1005", match_case=False, on_missing="ignore", verbose="ERROR")
        raw.pick("eeg")
        if raw.info["sfreq"] != sfreq:
            raw.resample(sfreq, verbose="ERROR")
        raw.filter(band_hz[0], hi, verbose="ERROR")
        if ch_names_ref is None:
            ch_names_ref = list(raw.ch_names)
        elif list(raw.ch_names) != ch_names_ref:
            print(f"  skip {path.name}: channel set mismatch")
            continue
        data = raw.get_data().astype(np.float32)  # (C, T_total) in volts
        n_epochs = data.shape[1] // T
        for i in range(n_epochs):
            epoch = data[:, i * T : (i + 1) * T]
            if np.any(np.isnan(epoch)) or np.any(np.std(epoch, axis=1) < 1e-9):
                continue
            epochs_list.append(epoch)
    if ch_names_ref is None:
        raise RuntimeError("No valid recordings preprocessed")
    print(f"  total epochs: {len(epochs_list)}, channels: {len(ch_names_ref)}")
    return epochs_list, ch_names_ref


def fit_bg_slope_global(
    epochs: list[np.ndarray],
    sfreq: float,
    fit_band_hz: tuple[float, float] = (2.0, 40.0),
) -> float:
    """Fit one global 1/f exponent beta from the sensor-level mean PSD.

    Avoids source localization (PhysioNet 10-10 names do not align with the
    dense 10-05 leadfield bank). With only a few hours of real EEG, a single
    cortex-wide beta is more robust than 994 noisy per-parcel estimates.
    """
    freqs = np.fft.rfftfreq(epochs[0].shape[1], d=1.0 / sfreq)
    band = (freqs >= fit_band_hz[0]) & (freqs <= fit_band_hz[1])
    psd_acc = np.zeros(int(band.sum()), dtype=np.float64)
    n_acc = 0
    for ep in epochs:
        psd = (np.abs(np.fft.rfft(ep, axis=1)) ** 2).mean(axis=0)
        psd_acc += psd[band]
        n_acc += 1
    psd_acc /= max(n_acc, 1)
    log_f = np.log(freqs[band])
    log_p = np.log(psd_acc + 1e-30)
    slope, _ = np.polyfit(log_f, log_p, 1)
    beta = float(-slope)
    print(f"  global 1/f beta = {beta:.3f} (band {fit_band_hz[0]}-{fit_band_hz[1]} Hz)")
    return beta


def fit_sensor_cov(
    epochs: list[np.ndarray],
    ch_names: list[str],
) -> np.ndarray:
    """Estimate the CxC inter-channel covariance averaged across epochs.

    Returns a (C, C) float32 matrix in the PhysioNet native channel order
    (use the companion ch_names list for downstream name-matching).
    """
    C = len(ch_names)
    cov_sum = np.zeros((C, C), dtype=np.float64)
    n_acc = 0
    for ep in epochs:
        ep = ep.astype(np.float64) - ep.astype(np.float64).mean(axis=1, keepdims=True)
        cov_sum += (ep @ ep.T) / ep.shape[1]
        n_acc += 1
    cov = cov_sum / max(n_acc, 1)
    diag = np.sqrt(np.diag(cov))
    corr = cov / (np.outer(diag, diag) + 1e-30)
    off = corr[~np.eye(C, dtype=bool)]
    print(f"  sensor cov ({C}x{C}): diag mean={float(np.mean(diag)):.3e}, "
          f"off-diag |corr| mean={float(np.mean(np.abs(off))):.3f}")
    return cov.astype(np.float32)


def main() -> int:
    args = _parse_args()
    cfg = GenerationConfig.from_yaml(args.config)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Calibration target: {args.out}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Config sfreq={cfg.temporal.sfreq}, window_s={cfg.temporal.window_s}")

    if not args.skip_download:
        print(f"\nDownloading PhysioNet EEGMMIDB ({args.n_subjects} subjects, runs {PHYSIONET_RUNS})...")
        raw_paths = download_physionet(args.cache_dir, args.n_subjects)
    else:
        raw_paths = sorted(args.cache_dir.rglob("*.edf"))
        print(f"\nUsing {len(raw_paths)} cached EDF files")
    if not raw_paths:
        raise RuntimeError("No PhysioNet recordings available")

    print("\nPreprocessing recordings...")
    epochs, ch_names = preprocess_recordings(
        raw_paths, cfg.temporal.sfreq, cfg.temporal.window_s
    )
    print(f"Got {len(epochs)} epochs of shape ({len(ch_names)}, "
          f"{int(cfg.temporal.sfreq * cfg.temporal.window_s)})")

    print("\nFitting global 1/f slope...")
    bg_slope_global = fit_bg_slope_global(epochs, cfg.temporal.sfreq)

    print("\nFitting sensor channel covariance...")
    sensor_cov = fit_sensor_cov(epochs, ch_names)

    print(f"\nWriting {args.out}")
    manifest = {
        "version": "physionet_v1",
        "source_dataset": "PhysioNet EEGMMIDB (Schalk 2004)",
        "runs_used": list(PHYSIONET_RUNS),
        "n_subjects_requested": args.n_subjects,
        "n_edf_files": len(raw_paths),
        "n_epochs": len(epochs),
        "sfreq": cfg.temporal.sfreq,
        "window_s": cfg.temporal.window_s,
        "fit_band_hz": [2.0, 40.0],
        "schema": "bg_slope_global=float; sensor_cov=(C,C); sensor_cov_ch_names=list[str]",
    }
    np.savez_compressed(
        args.out,
        _manifest_json=np.array(json.dumps(manifest)),
        bg_slope_global=np.array(bg_slope_global, dtype=np.float32),
        sensor_cov=sensor_cov,
        sensor_cov_ch_names=np.array(ch_names, dtype=object),
    )
    print(f"  wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
