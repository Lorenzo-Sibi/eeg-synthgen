from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import linregress

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}

# Nunez & Srinivasan 2006.
NORMATIVE_BACKGROUND_SLOPE_RANGE = (1.0, 2.0)

# Excludes [α, β] from the linregress fit; rhythmic peaks bias log-log slope
# (Donoghue et al. 2020, Nat Neurosci).
_APERIODIC_BANDS = ((1.0, 7.0), (35.0, 45.0))


def compute_psd(trace, sfreq):
    nperseg = min(len(trace), max(64, int(sfreq)))
    freqs, psd = sp_signal.welch(
        np.asarray(trace, dtype=np.float64),
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hann",
    )
    return freqs.astype(np.float32), psd.astype(np.float64)


def estimate_psd_slope(trace, sfreq, *, method="linregress"):
    freqs, psd = compute_psd(trace, sfreq)
    mask = np.zeros_like(freqs, dtype=bool)
    for lo, hi in _APERIODIC_BANDS:
        mask |= (freqs >= lo) & (freqs <= hi)
    mask &= psd > 0
    if mask.sum() < 5:
        return float("nan")
    if method == "fooof":
        try:
            from fooof import FOOOF

            fm = FOOOF(verbose=False)
            fm.fit(freqs.astype(np.float64), psd, freq_range=[1.0, 45.0])
            return float(fm.aperiodic_params_[1])
        except Exception:
            pass
    slope, *_ = linregress(np.log10(freqs[mask]), np.log10(psd[mask]))
    return float(-slope)


def compute_band_powers(trace, sfreq):
    freqs, psd = compute_psd(trace, sfreq)
    total_mask = (freqs >= 1.0) & (freqs <= 80.0)
    total = float(np.trapezoid(psd[total_mask], freqs[total_mask]))
    if total <= 1e-30:
        return {b: 0.0 for b in EEG_BANDS}
    return {
        b: float(
            np.trapezoid(
                psd[(freqs >= lo) & (freqs < hi)],
                freqs[(freqs >= lo) & (freqs < hi)],
            )
        )
        / total
        for b, (lo, hi) in EEG_BANDS.items()
    }
