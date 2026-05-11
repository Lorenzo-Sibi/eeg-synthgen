from __future__ import annotations

from typing import Any

import numpy as np

from synthgen.config import GenerationConfig, SEREEGABackendConfig
from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.base import SourceGeneratorBackend
from synthgen.sources.sereega_matlab import MatlabSereegaBridge


Signal = dict[str, Any]
Component = dict[str, Any]
_FAMILY_SIGNAL_CLASSES = {
    "erp": ("erp",),
    "oscillatory_burst": ("ersp",),
    "ar_correlated": ("arm",),
    "spike_interictal": ("erp",),
}

# Canonical source-level amplitudes (µV, pre-projection). Absolute amplitude is
# not a scientifically meaningful knob: the lead-field gain dominates the
# sensor-level scale, and SNR/SNIR is the invariant quantity (DeepSIF protocol).
_CANONICAL_AMPLITUDE_UV: float = 1.0
_CANONICAL_BACKGROUND_AMPLITUDE_UV: float = 0.1

def _signal(class_type: str, **params: Any) -> Signal:
    return {"type": class_type, "params": params}


def _sample_range(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    lo, hi = value_range
    return float(lo if lo == hi else rng.uniform(lo, hi))


def _sample_weighted_int(weights: dict[int, float], rng: np.random.Generator) -> int:
    keys = list(weights.keys())
    probs = np.array([weights[k] for k in keys], dtype=np.float64)
    return int(keys[int(rng.choice(len(keys), p=probs / probs.sum()))])


def _sample_erp_class(
    scenario: Scenario,
    source_idx: int,
    cfg: SEREEGABackendConfig,
    rng: np.random.Generator,
) -> Signal:
    onset_s = float(scenario.temporal_onsets_s[source_idx])
    if scenario.signal_family == "spike_interictal":
        spike_s = onset_s + _sample_range(rng, cfg.latency_jitter_s_range)
        amplitude = _CANONICAL_AMPLITUDE_UV
        return _signal(
            "erp",
            template="spike_interictal",
            peak_latencies_ms=[1000.0 * spike_s, 1000.0 * (spike_s + 0.04)],
            peak_widths_ms=[8.0, 60.0],
            peak_amplitudes=[-amplitude, 0.5 * amplitude],
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )

    n_peaks = _sample_weighted_int(cfg.erp_peak_count_weights, rng)
    latencies_s = [
        onset_s + _sample_range(rng, cfg.latency_jitter_s_range)
        for _ in range(n_peaks)
    ]
    widths_ms = [1000.0 * _sample_range(rng, cfg.erp_width_s_range) for _ in range(n_peaks)]
    amplitudes = [_CANONICAL_AMPLITUDE_UV for _ in range(n_peaks)]
    order = np.argsort(latencies_s)
    return _signal(
        "erp",
        peak_latencies_ms=[1000.0 * latencies_s[i] for i in order],
        peak_widths_ms=[widths_ms[i] for i in order],
        peak_amplitudes=[amplitudes[i] for i in order],
        seed=int(rng.integers(0, np.iinfo(np.int32).max)),
    )


def _sample_signal_class(
    class_name: str,
    scenario: Scenario,
    source_idx: int,
    cfg: SEREEGABackendConfig,
    rng: np.random.Generator,
) -> Signal:
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    if class_name == "erp":
        return _sample_erp_class(scenario, source_idx, cfg, rng)
    if class_name == "ersp":
        center_s = float(scenario.temporal_onsets_s[source_idx])
        center_s += _sample_range(rng, cfg.latency_jitter_s_range)
        return _signal(
            "ersp",
            frequency_hz=float(scenario.dominant_frequencies_hz[source_idx]),
            amplitude=_CANONICAL_AMPLITUDE_UV,
            phase_cycles=float(rng.uniform(0.0, 1.0)),
            mod_latency_ms=1000.0 * center_s,
            mod_width_ms=1000.0 * _sample_range(rng, cfg.burst_width_s_range),
            mod_taper=0.5,
            seed=seed,
        )
    if class_name == "noise":
        return _signal(
            "noise",
            color="pink",
            amplitude=_CANONICAL_AMPLITUDE_UV,
            seed=seed,
        )
    if class_name == "arm":
        return _signal(
            "arm",
            order=int(cfg.arm_order),
            amplitude=_CANONICAL_AMPLITUDE_UV,
            seed=seed,
        )
    raise ValueError(f"Unknown SEREEGA signal class: {class_name!r}")


def _signal_classes_for_family(scenario: Scenario) -> tuple[str, ...]:
    try:
        return _FAMILY_SIGNAL_CLASSES[scenario.signal_family]
    except KeyError as exc:
        raise ValueError(f"Unknown signal family: {scenario.signal_family!r}") from exc


def _scenario_components(
    scenario: Scenario,
    cfg: SEREEGABackendConfig | None = None,
    rng: np.random.Generator | None = None,
) -> list[Component]:
    cfg = cfg or SEREEGABackendConfig()
    rng = rng or np.random.default_rng(scenario.seed)
    n_sources = len(scenario.seed_vertex_indices)
    if len(scenario.dominant_frequencies_hz) < n_sources:
        raise ValueError("dominant_frequencies_hz must have one value per seed")
    if len(scenario.temporal_onsets_s) < n_sources:
        raise ValueError("temporal_onsets_s must have one value per seed")

    components: list[Component] = []
    class_names = _signal_classes_for_family(scenario)
    
    for source_idx, seed in enumerate(scenario.seed_vertex_indices):
        patch = (
            scenario.seed_patch_vertex_indices[source_idx]
            if source_idx < len(scenario.seed_patch_vertex_indices)
            and scenario.seed_patch_vertex_indices[source_idx]
            else [seed]
        )
        components.append(
            {
                "source": int(seed),
                "patch": list(dict.fromkeys([int(seed), *[int(v) for v in patch]])),
                "signal_classes": tuple(
                    _sample_signal_class(name, scenario, source_idx, cfg, rng)
                    for name in class_names
                ),
            }
        )
    return components


def _generate_component_activation(
    component: Component,
    T: int,
    sfreq: float,
    matlab_bridge: MatlabSereegaBridge,
) -> np.ndarray:
    activation = np.zeros(T, dtype=np.float32)
    for signal_class in component["signal_classes"]:
        activation += matlab_bridge.generate_signal(signal_class, T, sfreq)
    return activation.astype(np.float32)


def _apply_patch_spatial_weights(
    activation: np.ndarray,
    source_space: SourceSpace,
    component: Component,
    cfg: SEREEGABackendConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    patch = np.asarray(component["patch"], dtype=np.int64)
    if cfg.patch_spatial_profile == "uniform":
        weights = np.ones(len(patch), dtype=np.float32)
        sigma_mm = None
    elif cfg.patch_spatial_profile == "gaussian":
        distances_mm = np.linalg.norm(
            source_space.vertex_coords[patch] - source_space.vertex_coords[component["source"]],
            axis=1,
        )
        sigma_mm = _sample_range(rng, cfg.patch_spatial_sigma_mm_range)
        weights = np.exp(-0.5 * (distances_mm / sigma_mm) ** 2).astype(np.float32)
        weights = weights / (float(weights.max()) + 1e-8)
        
    # TODO: here it's possible to implement additional spatial profiles
    
    else:
        raise ValueError(f"Unknown patch spatial profile: {cfg.patch_spatial_profile!r}")

    return (
        (weights[:, np.newaxis] * activation[np.newaxis, :]).astype(np.float32),
        {
            "profile": cfg.patch_spatial_profile,
            "sigma_mm": sigma_mm,
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
        },
    )


def _generate_1f_background(
    N: int,
    T: int,
    sfreq: float,
    rng: np.random.Generator,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    freqs[0] = 1.0
    phases = rng.uniform(0.0, 2 * np.pi, size=(N, len(freqs)))
    bg = np.fft.irfft(np.exp(1j * phases) / np.sqrt(freqs), n=T).astype(np.float32)
    bg = bg / (np.std(bg, axis=1, keepdims=True) + 1e-8)
    return (bg * _CANONICAL_BACKGROUND_AMPLITUDE_UV).astype(np.float32)


class SEREEGABackend(SourceGeneratorBackend):
    """Source generator backed by the original MATLAB/SEREEGA toolbox."""

    def __init__(
        self,
        config: GenerationConfig,
        matlab_bridge: MatlabSereegaBridge | None = None,
    ) -> None:
        self._config = config
        if matlab_bridge is not None:
            self._matlab_bridge = matlab_bridge
        elif config.sereega.matlab_sereega_path is None:
            raise RuntimeError(
                "sereega.matlab_sereega_path is required because SEREEGA uses MATLAB."
            )
        else:
            self._matlab_bridge = MatlabSereegaBridge(config.sereega.matlab_sereega_path)

    def generate(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self._config.sereega
        T = self._config.temporal.n_samples_per_window # number of time points per sample
        N = len(source_space.vertex_coords) # number of sources in the source space (vertices)
        sfreq = self._config.temporal.sfreq
        source_activity = np.zeros((N, T), dtype=np.float32)

        scenario.sereega_trial_parameters = []

        for component in _scenario_components(scenario, cfg, rng):
            activation = _generate_component_activation(
                component, T, sfreq, self._matlab_bridge
            )
            weighted_activation, spatial_meta = _apply_patch_spatial_weights(
                activation, source_space, component, cfg, rng
            )
            source_activity[component["patch"]] += weighted_activation
            scenario.sereega_trial_parameters.append(
                {
                    "source": component["source"],
                    "patch": component["patch"],
                    "patch_size": len(component["patch"]),
                    "spatial": spatial_meta,
                    "signal_classes": [
                        {"type": signal["type"], "params": signal["params"]}
                        for signal in component["signal_classes"]
                    ],
                }
            )

        background_activity = _generate_1f_background(N, T, sfreq, rng)
        return source_activity, background_activity
