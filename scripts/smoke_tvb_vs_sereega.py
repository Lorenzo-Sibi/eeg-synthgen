#!/usr/bin/env python3
"""Compare TVB and SEREEGA synthetic-EEG generation on a tiny matched dataset.

Generates two small datasets (same n_samples, same global_seed, same anatomy
and montage, only the backend differs) and produces a multi-panel publication-
quality PDF + a JSON with KS / Wasserstein distances between the two
distributions on the most informative features.

Usage:
    python scripts/smoke_tvb_vs_sereega.py \\
        --config config/default.yaml --n-samples 5 \\
        --out reports/smoke_tvb_vs_sereega

Walltime: ~5 min/TVB-sample + ~0.05 s/SEREEGA-sample. Use n-samples=5 for a
~30-min smoke; n-samples=10-20 for stronger statistics if time allows.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

EEG_FEATURES = ("delta", "theta", "alpha", "beta", "gamma", "slope", "kurtosis", "rms")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("reports/smoke_tvb_vs_sereega"))
    p.add_argument("--global-seed", type=int, default=2026)
    return p.parse_args()


def _override_for_smoke(cfg, *, backend: str, n_samples: int, out_dir: Path, seed: int):
    return cfg.model_copy(update={
        "backend": backend,
        "n_samples": n_samples,
        "n_workers": 1,                      # both runs single-process for simplicity
        "global_seed": seed,
        "writer": cfg.writer.model_copy(update={
            "output_dir": out_dir,
            "chunk_size": max(1, n_samples),
        }),
    })


def generate_dataset(cfg, *, backend: str, n_samples: int, out_dir: Path, seed: int):
    from synthgen.pipeline_runner import PipelineRunner

    zarr_path = out_dir / "data.zarr"
    if zarr_path.exists() and any(zarr_path.iterdir()):
        print(f"[gen] backend={backend} -> {out_dir} already exists, skipping")
        return
    smoke_cfg = _override_for_smoke(cfg, backend=backend, n_samples=n_samples,
                                    out_dir=out_dir, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen] backend={backend} n={n_samples} -> {out_dir}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        PipelineRunner(smoke_cfg).run()


def load_eeg_groups(dataset_dir: Path) -> list[np.ndarray]:
    """Return a list of (N, C, T) EEG arrays, one per zarr group.

    Samples are grouped by (anatomy, montage) so different groups have
    different channel counts. Features are channel-independent and can
    be pooled across groups; raw arrays cannot be concatenated.
    """
    import zarr

    store = zarr.open(str(dataset_dir / "data.zarr"), mode="r")
    eeg_per_group = []
    for key in store.group_keys():
        eeg_per_group.append(store[key]["eeg"][:])          # (N, C, T)
    if not eeg_per_group:
        raise RuntimeError(f"No groups in {dataset_dir}/data.zarr")
    return eeg_per_group    


def compute_features(eeg_groups: list[np.ndarray], sfreq: float = 500.0) -> dict[str, np.ndarray]:
    """Compute features per (sample, channel), pooled across all groups."""
    from synthgen.analysis.spectral import compute_band_powers, estimate_psd_slope

    features: dict[str, list[float]] = {k: [] for k in EEG_FEATURES}
    for eeg in eeg_groups:
        N, C, T = eeg.shape
        for n in range(N):
            for c in range(C):
                trace = eeg[n, c]
                bp = compute_band_powers(trace, sfreq)
                for band in ("delta", "theta", "alpha", "beta", "gamma"):
                    features[band].append(float(bp[band]))
                slope = estimate_psd_slope(trace, sfreq)
                if not np.isnan(slope):
                    features["slope"].append(float(slope))
                x = trace.astype(np.float64)
                std = x.std() + 1e-12
                features["kurtosis"].append(float(np.mean(((x - x.mean()) / std) ** 4) - 3.0))
                features["rms"].append(float(np.sqrt(np.mean(x ** 2))))
    return {k: np.asarray(v, dtype=np.float64) for k, v in features.items()}


def compare_distributions(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, Any]:
    """KS-test + Wasserstein-1 between matched feature distributions."""
    from scipy.stats import ks_2samp, wasserstein_distance

    out: dict[str, Any] = {}
    for k in EEG_FEATURES:
        x = a.get(k, np.empty(0))
        y = b.get(k, np.empty(0))
        if x.size < 5 or y.size < 5:
            out[k] = {"skipped": True, "n_a": int(x.size), "n_b": int(y.size)}
            continue
        ks = ks_2samp(x, y)
        out[k] = {
            "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue),
            "wasserstein_1": float(wasserstein_distance(x, y)),
            "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "median_a": float(np.median(x)), "median_b": float(np.median(y)),
            "n_a": int(x.size), "n_b": int(y.size),
        }
    return out


def load_target_bg_sensor_rms(cfg) -> float | None:
    """Re-load the reference value from banks/noise/<calibration_id>.npz
    so we can show it in the smoke report. Mirrors
    AcquisitionPipeline._load_reference_sensor_rms.
    """
    cid = cfg.noise.calibration_id
    if cid is None:
        return None
    p = Path("banks/noise") / f"{cid}.npz"
    if not p.exists():
        return None
    b = np.load(p, allow_pickle=True)
    if "sensor_cov" not in b.files:
        return None
    diag = np.diag(np.asarray(b["sensor_cov"], dtype=np.float64))
    diag = diag[diag > 0]
    return float(np.sqrt(np.median(diag))) if diag.size else None


def compute_verdict(cmp_: dict[str, Any], target_rms: float | None) -> tuple[bool, dict[str, Any]]:
    """Pass/fail gates for the cross-backend normalization design.

    A: amplitude aligned — sensor RMS within 2x and distributions not disjoint.
    B: diversity preserved — at least 2 spectral/statistical features remain
       significantly different (KS p<0.05).
    C: reference loaded — target_bg_sensor_rms is a positive float (the
       PhysioNet calibration is reachable at pipeline init time).
    """
    out: dict[str, Any] = {}

    # Gate A: amplitude alignment
    r = cmp_.get("rms", {})
    if not r.get("skipped"):
        ratio = float(r["mean_a"]) / max(float(r["mean_b"]), 1e-30)
        out["A_amplitude_aligned"] = {
            "pass": bool(0.5 <= ratio <= 2.0),
            "rms_ratio_tvb_sereega": ratio,
            "tvb_rms_mean": float(r["mean_a"]),
            "sereega_rms_mean": float(r["mean_b"]),
            "ks_p": float(r["ks_p"]),
            "description": "TVB and SEREEGA sensor RMS within a factor of 2",
        }
    else:
        out["A_amplitude_aligned"] = {"pass": False, "reason": "rms comparison skipped"}

    # Gate B: spectral/statistical diversity preserved
    diverse = [
        k for k in ("slope", "kurtosis", "delta", "alpha", "beta", "gamma")
        if not cmp_.get(k, {}).get("skipped") and cmp_[k]["ks_p"] < 0.05
    ]
    out["B_diversity_preserved"] = {
        "pass": len(diverse) >= 2,
        "distinguishing_features": diverse,
        "description": "At least 2 features still differ between backends (KS p<0.05)",
    }

    # Gate C: reference loaded from calibration bank
    out["C_reference_loaded"] = {
        "pass": target_rms is not None and target_rms > 0.0,
        "target_bg_sensor_rms_V": target_rms,
        "target_bg_sensor_rms_uV": (target_rms * 1e6) if target_rms else None,
        "description": "PhysioNet reference reachable from banks/noise/<id>.npz",
    }

    overall = all(bool(g.get("pass", False)) for g in out.values())
    return overall, out


def make_figure(feats_tvb: dict[str, np.ndarray], feats_sereega: dict[str, np.ndarray],
                eeg_tvb_groups: list[np.ndarray], eeg_sereega_groups: list[np.ndarray],
                cmp_: dict[str, Any], out_path: Path, sfreq: float = 500.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # A) PSD average across all (sample, channel) — pool across groups
    def _avg_psd(eeg_groups):
        T = eeg_groups[0].shape[2]
        f = np.fft.rfftfreq(T, d=1.0 / sfreq)
        psd_sum = np.zeros(f.size, dtype=np.float64)
        n_total = 0
        for eeg in eeg_groups:
            n = eeg.shape[0] * eeg.shape[1]
            psd_sum += (np.abs(np.fft.rfft(eeg, axis=2)) ** 2).sum(axis=(0, 1))
            n_total += n
        return f, psd_sum / max(n_total, 1)
    ax = axes[0, 0]
    n_tvb = sum(e.shape[0] for e in eeg_tvb_groups)
    n_se  = sum(e.shape[0] for e in eeg_sereega_groups)
    f, p = _avg_psd(eeg_tvb_groups);     ax.loglog(f[1:], p[1:], label=f"TVB (n={n_tvb})", lw=2)
    f, p = _avg_psd(eeg_sereega_groups); ax.loglog(f[1:], p[1:], label=f"SEREEGA (n={n_se})", lw=2)
    ax.set_xlabel("Hz"); ax.set_ylabel("PSD"); ax.set_title("(A) Sensor PSD (mean across samples,ch)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # B) Histogram of slope
    ax = axes[0, 1]
    bins = np.linspace(-4, 4, 30)
    ax.hist(feats_tvb["slope"], bins=bins, alpha=0.55, label="TVB", density=True)
    ax.hist(feats_sereega["slope"], bins=bins, alpha=0.55, label="SEREEGA", density=True)
    ks = cmp_["slope"]
    ax.set_xlabel("1/f slope (5-40 Hz)"); ax.set_ylabel("density")
    ax.set_title(f"(B) PSD slope — KS={ks['ks_stat']:.2f}, W1={ks['wasserstein_1']:.2f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    # C) Kurtosis distributions
    ax = axes[0, 2]
    bins = np.linspace(-3, 30, 30)
    ax.hist(np.clip(feats_tvb["kurtosis"], -3, 30), bins=bins, alpha=0.55, label="TVB", density=True)
    ax.hist(np.clip(feats_sereega["kurtosis"], -3, 30), bins=bins, alpha=0.55, label="SEREEGA", density=True)
    ks = cmp_["kurtosis"]
    ax.set_xlabel("excess kurtosis"); ax.set_ylabel("density")
    ax.set_title(f"(C) Kurtosis — KS={ks['ks_stat']:.2f}, W1={ks['wasserstein_1']:.2f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    # D) RMS distributions
    ax = axes[1, 0]
    ax.hist(np.log10(feats_tvb["rms"] + 1e-30), bins=30, alpha=0.55, label="TVB", density=True)
    ax.hist(np.log10(feats_sereega["rms"] + 1e-30), bins=30, alpha=0.55, label="SEREEGA", density=True)
    ks = cmp_["rms"]
    ax.set_xlabel("log10(sensor RMS)"); ax.set_ylabel("density")
    ax.set_title(f"(D) RMS — KS={ks['ks_stat']:.2f}, W1={ks['wasserstein_1']:.2f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    # E) Band powers (mean per band)
    ax = axes[1, 1]
    bands = ("delta", "theta", "alpha", "beta", "gamma")
    means_tvb = [feats_tvb[b].mean() for b in bands]
    means_se = [feats_sereega[b].mean() for b in bands]
    x = np.arange(len(bands)); w = 0.35
    ax.bar(x - w / 2, means_tvb, w, label="TVB")
    ax.bar(x + w / 2, means_se, w, label="SEREEGA")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel("mean band power"); ax.set_title("(E) Mean band power")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # F) Sample trace (1 channel) overlay — use first group of each
    ax = axes[1, 2]
    e_tvb_first = eeg_tvb_groups[0]
    e_se_first = eeg_sereega_groups[0]
    t = np.arange(e_tvb_first.shape[2]) / sfreq
    ax.plot(t, e_tvb_first[0, 0], label="TVB sample 0 ch 0", lw=1.0)
    ax.plot(t, e_se_first[0, 0], label="SEREEGA sample 0 ch 0", lw=1.0, alpha=0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("EEG"); ax.set_title("(F) Sample trace")
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"TVB vs SEREEGA smoke comparison (TVB n={n_tvb}, SEREEGA n={n_se})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from synthgen.config import GenerationConfig
    cfg = GenerationConfig.from_yaml(args.config)

    tvb_dir = args.out / "ds_tvb"
    se_dir = args.out / "ds_sereega"

    generate_dataset(cfg, backend="tvb", n_samples=args.n_samples,
                     out_dir=tvb_dir, seed=args.global_seed)
    generate_dataset(cfg, backend="sereega", n_samples=args.n_samples,
                     out_dir=se_dir, seed=args.global_seed)

    print("[load] decoding zarr datasets")
    eeg_tvb_groups = load_eeg_groups(tvb_dir)
    eeg_se_groups = load_eeg_groups(se_dir)
    for i, e in enumerate(eeg_tvb_groups):
        print(f"  TVB     group {i}: shape {e.shape}")
    for i, e in enumerate(eeg_se_groups):
        print(f"  SEREEGA group {i}: shape {e.shape}")

    print("[features] computing spectral + statistical features")
    feat_tvb = compute_features(eeg_tvb_groups, sfreq=cfg.temporal.sfreq)
    feat_se = compute_features(eeg_se_groups, sfreq=cfg.temporal.sfreq)

    print("[stats] KS + Wasserstein per feature")
    cmp_ = compare_distributions(feat_tvb, feat_se)
    (args.out / "comparison.json").write_text(json.dumps(cmp_, indent=2))

    target_rms = load_target_bg_sensor_rms(cfg)
    overall_pass, verdict = compute_verdict(cmp_, target_rms)
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"[verdict] overall: {'PASS' if overall_pass else 'FAIL'}")
    for name, g in verdict.items():
        print(f"  {name}: {'PASS' if g.get('pass') else 'FAIL'}")

    summary_lines = ["# TVB vs SEREEGA smoke comparison\n",
                     f"- n_samples per backend: {args.n_samples}",
                     f"- TVB groups: {[e.shape for e in eeg_tvb_groups]}",
                     f"- SEREEGA groups: {[e.shape for e in eeg_se_groups]}",
                     f"- Reference target_bg_sensor_rms: "
                     f"{(target_rms * 1e6):.2f} µV" if target_rms else "- Reference target_bg_sensor_rms: (none)",
                     "",
                     "## Verdict",
                     f"**Overall:** {'PASS' if overall_pass else 'FAIL'}",
                     ""]
    for name, g in verdict.items():
        ok = "PASS" if g.get("pass") else "FAIL"
        detail_parts = [f"{k} = {v}" for k, v in g.items() if k not in ("pass", "description")]
        summary_lines.append(f"- **{name}**: {ok} — {g.get('description', '')}")
        if detail_parts:
            summary_lines.append(f"  - " + "; ".join(detail_parts))
    summary_lines.append("")

    summary_lines.extend([
        "## Feature distributions",
        "",
        "| Feature | TVB mean | SEREEGA mean | KS stat | KS p | W1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for k in EEG_FEATURES:
        r = cmp_[k]
        if r.get("skipped"):
            summary_lines.append(f"| {k} | — | — | — | — | — |")
            continue
        summary_lines.append(
            f"| {k} | {r['mean_a']:.3e} | {r['mean_b']:.3e} | "
            f"{r['ks_stat']:.3f} | {r['ks_p']:.2e} | {r['wasserstein_1']:.3e} |"
        )
    (args.out / "summary.md").write_text("\n".join(summary_lines))

    print(f"[plot] writing {args.out}/figure.pdf and figure.png")
    make_figure(feat_tvb, feat_se, eeg_tvb_groups, eeg_se_groups, cmp_,
                args.out / "figure.pdf", sfreq=cfg.temporal.sfreq)
    make_figure(feat_tvb, feat_se, eeg_tvb_groups, eeg_se_groups, cmp_,
                args.out / "figure.png", sfreq=cfg.temporal.sfreq)
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
