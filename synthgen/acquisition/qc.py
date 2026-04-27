from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from synthgen.config import QCConfig
from synthgen.sample import EEGSample


@dataclass
class QCResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def check_sample(sample: EEGSample, config: QCConfig) -> QCResult:
    reasons: list[str] = []

    # 0. Numerical integrity: reject NaN/Inf before any further checks
    if not np.all(np.isfinite(sample.eeg)):
        reasons.append("EEG contains NaN or Inf values")

    # 1. Minimum valid channels: channels with std above float32 noise floor
    channel_stds = np.std(sample.eeg, axis=1)
    valid_channels = int(np.sum(channel_stds > 1e-6))
    if valid_channels < config.min_valid_channels:
        reasons.append(
            f"only {valid_channels} valid channels (minimum {config.min_valid_channels})"
        )

    # 2. Minimum inter-source distance
    if sample.params.inter_source_distances_mm:
        min_dist = float(min(sample.params.inter_source_distances_mm))
        if min_dist < config.min_inter_source_distance_mm:
            reasons.append(
                f"minimum inter-source distance {min_dist:.1f} mm is below "
                f"threshold {config.min_inter_source_distance_mm:.1f} mm"
            )

    return QCResult(passed=len(reasons) == 0, reasons=reasons)
