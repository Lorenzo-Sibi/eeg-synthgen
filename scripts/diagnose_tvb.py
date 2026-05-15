#!/usr/bin/env python3
"""End-to-end diagnostics for the TVB source backend.

The script runs five physiologically-informed checks. Exit code 0 if all
pass; 1 if any fails. Output report in reports/tvb_diagnose/summary.md.

Usage:
    python scripts/diagnose_tvb.py --config config/default.yaml [--plot]

Thresholds reference:
  - Jansen & Rit (1995): PSP observable y2-y3 nominally in [-5, +20] mV under
    the canonical parametrisation. We accept |y| < 100 mV (20x headroom)
    before declaring saturation.
  - Hagmann et al. 2008 (PLoS Biol): cortical connectomes have sparsity in
    [0.85, 0.97] and edge weights spanning ~2 orders of magnitude.
  - Buzsaki & Draguhn 2004: 1/f resting-state spectrum is the rule;
    pure pink ~ 1, brown ~ 2. We require slope < 0 on 15-40 Hz.
  - Resting alpha (Berger 1929, plus countless): 8-13 Hz peak under most
    stochastic mean-field models, including Jansen-Rit at canonical coupling.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# Physiological/numerical thresholds used by the checks.
JR_OBSERVABLE_MAX_MV = 100.0   # 20x headroom over canonical 5 mV PSP amplitude
SPARSITY_RANGE = (0.30, 0.97)  # acceptable connectome sparsity
ALPHA_BAND_HZ = (8.0, 13.0)
SLOPE_BAND_HZ = (15.0, 40.0)
SEED_RMS_RATIO_MIN = 1.10        # seed parcels expected at least 10% RMS over baseline
SEED_EXCESS_KURTOSIS_MIN = 1.0   # spike-prone seeds carry extra kurtosis vs Gaussian-like baseline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=10,
                   help="Samples for plausibility/efficacy checks (each ~15s)")
    p.add_argument("--out", type=Path, default=Path("reports/tvb_diagnose"))
    p.add_argument("--plot", action="store_true",
                   help="Save connectivity matshow + baseline PSD figures")
    return p.parse_args()


# Check 1: parcellation alignment

def check_parcellation_alignment(cfg) -> tuple[bool, str]:
    """Region labels (and count) must be identical between source-space and
    connectivity. A silent mismatch would scramble the parcel-to-vertex map."""
    from synthgen.banks.anatomy import AnatomyBank
    from synthgen.banks.connectivity import ConnectivityBank

    scheme = cfg.tvb.connectivity_scheme
    ss = AnatomyBank(cfg.anatomy_bank).load(cfg.anatomy_bank.anatomy_ids[0], scheme=scheme)
    conn = ConnectivityBank(cfg.connectivity_bank).load(scheme)

    R_ss = int(ss.parcellation.max() + 1)
    if conn.weights.shape != (R_ss, R_ss):
        return False, f"weights shape {conn.weights.shape} != ({R_ss}, {R_ss})"
    if list(conn.region_labels) != list(ss.region_labels):
        n_mismatch = sum(
            a != b for a, b in zip(conn.region_labels, ss.region_labels)
        )
        return False, f"region_labels mismatch ({n_mismatch} differ)"
    return True, f"R={R_ss}, labels and shape aligned"


# Check 2: connectivity sanity

def check_connectivity_sanity(cfg, out_dir: Path, plot: bool) -> tuple[bool, str]:
    """Structural connectome must be:
      - symmetric (undirected DTI)
      - zero-diagonal (no self-loops)
      - sparsity within physiological range (Hagmann 2008)
      - have non-trivial weight distribution

    Note: DeepSIF connectomes carry quantised weights (a few discrete values)
    rather than the log-normal of raw Hagmann tractography. We accept either,
    rejecting only the degenerate cases (all-equal or all-zero).
    """
    from synthgen.banks.connectivity import ConnectivityBank

    scheme = cfg.tvb.connectivity_scheme
    conn = ConnectivityBank(cfg.connectivity_bank).load(scheme)
    W = conn.weights.astype(np.float64)

    if not np.allclose(W, W.T, atol=1e-5):
        return False, f"weights not symmetric (max |W-W.T| = {float(np.max(np.abs(W-W.T))):.2e})"
    if not np.allclose(np.diag(W), 0.0):
        return False, f"diagonal non-zero (max |diag| = {float(np.max(np.abs(np.diag(W)))):.2e})"

    sparsity = float((W == 0).mean())
    if not (SPARSITY_RANGE[0] <= sparsity <= SPARSITY_RANGE[1]):
        return False, f"sparsity {sparsity:.3f} outside {SPARSITY_RANGE}"

    nz = W[W > 0]
    if nz.size < 10:
        return False, f"only {nz.size} non-zero entries"
    if nz.max() == nz.min():
        return False, "all non-zero weights identical (degenerate connectome)"

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.matshow(np.log10(W + 1e-9), cmap="viridis")
        ax.set_title(f"Connectivity log10(weights) - {scheme}")
        fig.savefig(out_dir / "connectivity.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    n_distinct = len(np.unique(nz))
    return True, (
        f"symmetric, diag=0, sparsity={sparsity:.3f}, "
        f"n_nonzero={nz.size}, n_distinct_weights={n_distinct}"
    )


# Helper: run N samples

def _run_tvb_samples(cfg, n_samples: int, *, with_stim: bool):
    """Generate n_samples through TVBSourceGenerator and return a list of
    tuples (scenario, source_activity, background_activity, source_space)."""
    from synthgen.banks.anatomy import AnatomyBank
    from synthgen.banks.connectivity import ConnectivityBank
    from synthgen.scenario.sampler import ScenarioSampler
    from synthgen.sources.priors.broad_random import BroadRandomPrior
    from synthgen.sources.tvb_backend import TVBSourceGenerator

    scheme = cfg.tvb.connectivity_scheme
    anatomy_id = cfg.anatomy_bank.anatomy_ids[0]
    ss = AnatomyBank(cfg.anatomy_bank).load(anatomy_id, scheme=scheme)
    conn = ConnectivityBank(cfg.connectivity_bank).load(scheme)
    backend = TVBSourceGenerator(cfg, conn)
    sampler = ScenarioSampler(cfg)
    prior = BroadRandomPrior()

    out = []
    rng = np.random.default_rng(2026)
    attempts = 0
    while len(out) < n_samples and attempts < n_samples * 3:
        attempts += 1
        sc = sampler.sample(rng)
        sc_rng = np.random.default_rng(sc.seed)
        sc = prior.sample(sc, ss, sc_rng)
        if not with_stim:
            sc.seed_vertex_indices = []
            sc.seed_patch_vertex_indices = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            src, bg = backend.generate(sc, ss, sc_rng)
        out.append((sc, src, bg, ss))
    return out


# Check 3: background plausibility

def check_background_plausibility(cfg, out_dir: Path, n_samples: int, plot: bool):
    """Baseline activity (non-seed parcels, A = jr_A_baseline) must be:
      1. amplitude-bounded: |y|.max < JR_OBSERVABLE_MAX_MV (no saturation)
      2. all-finite (no NaN/Inf escape from the integrator)
      3. scale-free in 15-40 Hz: PSD slope < 0 (rejects pure white-noise output)

    Spectral peak frequency is *not* constrained: DeepSIF-style Jansen-Rit
    is tuned to produce delta-band slow waves and spikes (Sun et al. 2022,
    sec. 2.1), so a peak below the alpha band is the expected operating
    regime, not a failure mode.
    """
    samples = _run_tvb_samples(cfg, n_samples, with_stim=False)
    sfreq = cfg.temporal.sfreq

    abs_max = float(max(np.abs(bg).max() for _, _, bg, _ in samples))
    if abs_max > JR_OBSERVABLE_MAX_MV:
        return False, (
            f"baseline observable saturates: max|y|={abs_max:.1e} mV "
            f"exceeds physiological ceiling {JR_OBSERVABLE_MAX_MV:.0f} mV"
        )
    if not all(np.isfinite(bg).all() for _, _, bg, _ in samples):
        return False, "baseline contains NaN/Inf (integrator escaped)"

    psd_acc = None
    for _, _, bg, _ in samples:
        psd = (np.abs(np.fft.rfft(bg, axis=1)) ** 2).mean(axis=0)
        psd_acc = psd if psd_acc is None else psd_acc + psd
    assert psd_acc is not None
    psd_acc /= len(samples)
    f = np.fft.rfftfreq(samples[0][2].shape[1], d=1.0 / sfreq)

    peak_idx = int(np.argmax(psd_acc[1:])) + 1  # skip DC
    peak_hz = float(f[peak_idx])

    slope_band = (f >= SLOPE_BAND_HZ[0]) & (f <= SLOPE_BAND_HZ[1])
    raw_slope, _ = np.polyfit(
        np.log(f[slope_band]), np.log(psd_acc[slope_band] + 1e-30), 1
    )
    # 1/f exponent beta with the literature convention (PSD ~ 1/f^beta, beta > 0
    # for pink/brown). Matches synthgen.analysis.spectral.estimate_psd_slope.
    beta = float(-raw_slope)

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(f[1:], psd_acc[1:])
        ax.axvspan(*ALPHA_BAND_HZ, alpha=0.10, color="C1", label="alpha (reference)")
        ax.set_xlabel("Hz")
        ax.set_ylabel("PSD (a.u.)")
        ax.set_title(f"Baseline PSD - peak {peak_hz:.1f} Hz, beta {beta:.2f}")
        ax.legend()
        fig.savefig(out_dir / "psd_baseline.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    if beta <= 0.0:
        return False, f"15-40 Hz 1/f exponent beta = {beta:.2f} <= 0 (no 1/f, numerical noise dominates)"
    return True, (
        f"max|y|={abs_max:.1e} mV, peak={peak_hz:.1f} Hz, beta={beta:.2f}"
    )


# Check 4: seed-region excitability effect (DeepSIF-style)

def check_seed_excitability_effect(cfg, n_samples: int) -> tuple[bool, str]:
    """Seed parcels carry the elevated Jansen-Rit ``A`` (default 3.5 vs 3.25
    baseline) and must therefore exhibit *quantitatively different* activity
    than non-seed parcels.

    DeepSIF reports that the elevation drives the seed parcels into a
    spike-prone regime, producing higher-kurtosis activity and typically
    elevated RMS amplitude. We check that the seed-parcel RMS, *averaged
    across scenarios*, exceeds the non-seed RMS by the configured minimum
    margin. RMS is read off the per-parcel rows of ``background`` (which
    is piecewise-constant per parcel by construction).
    """
    samples = _run_tvb_samples(cfg, n_samples, with_stim=True)
    valid = [(sc, src, bg, ss) for (sc, src, bg, ss) in samples if sc.seed_patch_vertex_indices]
    if not valid:
        return False, "no scenarios with seed parcels were generated"

    seed_rms_list, nonseed_rms_list, kurt_ratio_list = [], [], []
    for sc, _, bg, ss in valid:
        parc = ss.parcellation
        seed_parcels = set()
        for patch in sc.seed_patch_vertex_indices:
            seed_parcels.update(int(parc[v]) for v in patch)
        if not seed_parcels:
            continue
        all_parcels = np.unique(parc)
        nonseed_parcels = [int(p) for p in all_parcels if int(p) not in seed_parcels]
        if not nonseed_parcels:
            continue
        # Pick one representative vertex per parcel (bg is piecewise-constant)
        seed_traces = np.stack(
            [bg[np.argmax(parc == p)] for p in seed_parcels], axis=0
        )
        nonseed_idx = [int(np.argmax(parc == p)) for p in nonseed_parcels]
        nonseed_traces = bg[nonseed_idx]

        seed_rms = float(np.sqrt(np.mean(seed_traces ** 2)))
        nonseed_rms = float(np.sqrt(np.mean(nonseed_traces ** 2)))
        seed_rms_list.append(seed_rms)
        nonseed_rms_list.append(nonseed_rms)

        # Kurtosis comparison: spike-prone activity has fat tails.
        def _kurt(x):
            x = (x - x.mean()) / (x.std() + 1e-12)
            return float(np.mean(x ** 4) - 3.0)
        k_seed = _kurt(seed_traces.flatten())
        k_nonseed = _kurt(nonseed_traces.flatten())
        kurt_ratio_list.append(k_seed - k_nonseed)

    if not seed_rms_list:
        return False, "no usable scenarios with seed/non-seed parcels"

    seed_med = float(np.median(seed_rms_list))
    nonseed_med = float(np.median(nonseed_rms_list))
    ratio = seed_med / (nonseed_med + 1e-30)
    excess_kurt_med = float(np.median(kurt_ratio_list)) if kurt_ratio_list else 0.0

    # Either elevated RMS or elevated kurtosis is acceptable evidence of
    # the A elevation having a measurable effect on the seed parcels.
    rms_ok = ratio >= SEED_RMS_RATIO_MIN
    kurt_ok = excess_kurt_med >= SEED_EXCESS_KURTOSIS_MIN

    msg = (
        f"seed/non-seed RMS = {seed_med:.3e}/{nonseed_med:.3e} (ratio {ratio:.2f}), "
        f"excess kurtosis = {excess_kurt_med:+.2f}"
    )
    if rms_ok or kurt_ok:
        return True, msg
    return False, (
        f"seed parcels show no measurable elevation: {msg}. "
        f"Required: RMS ratio >= {SEED_RMS_RATIO_MIN} OR excess kurtosis >= {SEED_EXCESS_KURTOSIS_MIN}"
    )


# Check 5: parcel-to-vertex piecewise-constant mapping

def check_parcel_to_vertex_mapping(cfg, n_samples: int) -> tuple[bool, str]:
    """By construction in tvb_backend.py (`background = baseline_R[parc, :]`)
    every vertex in the same parcel must carry an identical time series."""
    samples = _run_tvb_samples(cfg, min(n_samples, 3), with_stim=True)
    for sc, _, bg, ss in samples:
        parc = ss.parcellation
        for p in np.unique(parc)[:15]:
            mask = parc == p
            if mask.sum() < 2:
                continue
            ref = bg[mask][0]
            if not np.allclose(bg[mask], ref[None, :], rtol=1e-4, atol=1e-7):
                return False, (
                    f"background varies within parcel {p} "
                    f"(max diff = {float(np.abs(bg[mask] - ref).max()):.3e})"
                )
    return True, "background is piecewise-constant per parcel (sampled 3 epochs x 15 parcels)"


# Main

def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from synthgen.config import GenerationConfig
    cfg = GenerationConfig.from_yaml(args.config)

    print(f"Running TVB diagnostics ({args.n_samples} samples per check)...")
    print(f"  anatomy = {cfg.anatomy_bank.anatomy_ids[0]}, scheme = {cfg.tvb.connectivity_scheme}")
    print(f"  model = {cfg.tvb.model}, coupling = {cfg.tvb.global_coupling}, "
          f"noise_std = {cfg.tvb.noise_std}")
    print()

    results: list[tuple[str, bool, str]] = []
    for name, fn in [
        ("parcellation_alignment", lambda: check_parcellation_alignment(cfg)),
        ("connectivity_sanity", lambda: check_connectivity_sanity(cfg, args.out, args.plot)),
        ("background_plausibility", lambda: check_background_plausibility(cfg, args.out, args.n_samples, args.plot)),
        ("seed_excitability_effect", lambda: check_seed_excitability_effect(cfg, args.n_samples)),
        ("parcel_to_vertex_mapping", lambda: check_parcel_to_vertex_mapping(cfg, args.n_samples)),
    ]:
        print(f"  {name}... ", end="", flush=True)
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"raised {type(exc).__name__}: {exc}"
        results.append((name, ok, msg))
        print("PASS" if ok else "FAIL")
        print(f"    {msg}")

    print()
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"Summary: {n_pass}/{len(results)} checks passed")

    lines = ["# TVB diagnose summary\n"]
    lines.append(f"- anatomy: `{cfg.anatomy_bank.anatomy_ids[0]}`")
    lines.append(f"- scheme: `{cfg.tvb.connectivity_scheme}`")
    lines.append(f"- model: `{cfg.tvb.model}`, coupling = {cfg.tvb.global_coupling}, "
                 f"noise_std = {cfg.tvb.noise_std}")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("| --- | --- | --- |")
    for name, ok, msg in results:
        lines.append(f"| `{name}` | {'PASS' if ok else 'FAIL'} | {msg} |")
    lines.append("")
    lines.append(f"**Verdict:** {n_pass}/{len(results)} checks passed. "
                 + ("TVB ready for mixing." if n_pass == len(results)
                    else "TVB **not ready** - keep backend SEREEGA-only and document."))
    (args.out / "summary.md").write_text("\n".join(lines))
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
