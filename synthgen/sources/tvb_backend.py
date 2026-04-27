from __future__ import annotations

from math import gcd
from typing import Any

import numpy as np

from synthgen.banks.connectivity import Connectivity
from synthgen.config import GenerationConfig
from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.base import SourceGeneratorBackend
from synthgen.sources.tvb_models import get_tvb_model
from synthgen.sources.tvb_stimulus import build_stimulus_pattern, seed_parcel_ids


def _require_tvb() -> None:
    try:
        import tvb  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "TVB backend requested but tvb-library is not installed. "
            'Install with `pip install -e ".[tvb]"`.'
        ) from e


def _build_tvb_connectivity(conn: Connectivity, speed_mm_per_ms: float) -> Any:
    from tvb.datatypes.connectivity import Connectivity as TVBConnectivity

    tvb_conn = TVBConnectivity(
        weights=conn.weights.astype(np.float64),
        tract_lengths=conn.tract_lengths.astype(np.float64),
        centres=conn.region_centers.astype(np.float64),
        region_labels=np.array(conn.region_labels),
        speed=np.array([speed_mm_per_ms]),
    )
    tvb_conn.configure()
    return tvb_conn


def _build_simulator(cfg_tvb, tvb_conn, stim=None):
    from tvb.simulator import coupling, integrators, monitors, noise, simulator

    model = get_tvb_model(cfg_tvb.model)
    cpl = coupling.Linear(a=np.array([cfg_tvb.global_coupling]))
    integ = integrators.HeunStochastic(
        dt=cfg_tvb.integrator_dt_ms,
        noise=noise.Additive(nsig=np.array([cfg_tvb.noise_sigma])),
    )
    mon = monitors.TemporalAverage(period=cfg_tvb.integrator_dt_ms)
    sim = simulator.Simulator(
        model=model,
        connectivity=tvb_conn,
        coupling=cpl,
        integrator=integ,
        monitors=(mon,),
        stimulus=stim,
    )
    return sim


def _extract_observable(raw_state: np.ndarray, model_name: str) -> np.ndarray:
    """raw_state shape: (T, state_vars, n_regions, modes) -> (n_regions, T)."""
    if model_name == "jansen_rit":
        y = raw_state[:, 1, :, 0] - raw_state[:, 2, :, 0]
    elif model_name == "generic_2d_oscillator":
        y = raw_state[:, 0, :, 0]
    elif model_name == "wilson_cowan":
        y = raw_state[:, 0, :, 0]
    else:
        raise ValueError(f"Unknown model {model_name!r}")
    return y.T.astype(np.float32)


def _resample_to(
    y_region: np.ndarray,
    sfreq_src_hz: float,
    sfreq_dst_hz: float,
    T_out: int,
) -> np.ndarray:
    from scipy.signal import resample_poly

    if abs(sfreq_src_hz - sfreq_dst_hz) < 1e-6:
        out = y_region
    else:
        num = int(round(sfreq_dst_hz))
        den = int(round(sfreq_src_hz))
        g = gcd(num, den)
        out = resample_poly(y_region, up=num // g, down=den // g, axis=1)
    if out.shape[1] > T_out:
        out = out[:, :T_out]
    elif out.shape[1] < T_out:
        pad = np.zeros((out.shape[0], T_out - out.shape[1]), dtype=out.dtype)
        out = np.concatenate([out, pad], axis=1)
    return out.astype(np.float32)


class TVBSourceGenerator(SourceGeneratorBackend):
    """TVB-backed generator for (source_activity, background_activity).

    Both components are derived from paired TVB NMM simulations on the
    scenario's parcellation. The source term is isolated via
    with-vs-without-stimulus subtraction on seed parcels.
    """

    def __init__(
        self,
        config: GenerationConfig,
        connectivity: Connectivity,
    ) -> None:
        _require_tvb()
        self._cfg = config
        self._cfg_tvb = config.tvb
        self._conn = connectivity
        self._tvb_conn = _build_tvb_connectivity(
            connectivity, self._cfg_tvb.conduction_speed_mm_per_ms
        )

    def _make_stim(
        self,
        weights_R: np.ndarray,
        temporal_pattern: np.ndarray,
        dt_ms: float,
    ) -> Any:
        from tvb.basic.neotraits.api import Attr
        from tvb.datatypes.equations import TemporalApplicableEquation
        from tvb.datatypes.patterns import StimuliRegion

        class _Pattern(TemporalApplicableEquation):
            equation = Attr(
                field_type=str,
                label="External",
                default="externally_provided",
                doc="externally provided discrete pattern sampled at dt_ms",
            )

            def __init__(self, vals: np.ndarray, step_ms: float) -> None:
                super().__init__()
                self._vals = np.asarray(vals, dtype=np.float64)
                self._step_ms = step_ms

            def evaluate(self, var):
                t_ms = np.asarray(var, dtype=np.float64)
                idx = np.clip(
                    (t_ms / self._step_ms).astype(int), 0, len(self._vals) - 1
                )
                return self._vals[idx]

        eq = _Pattern(temporal_pattern, dt_ms)
        stim = StimuliRegion(
            connectivity=self._tvb_conn,
            weight=weights_R.astype(np.float64),
            temporal=eq,
        )
        stim.configure_space()
        return stim

    def generate(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if source_space.parcellation_scheme != self._conn.scheme:
            raise ValueError(
                f"parcellation_scheme mismatch: source_space={source_space.parcellation_scheme!r} "
                f"connectivity={self._conn.scheme!r}"
            )

        parc = source_space.parcellation
        N = parc.shape[0]
        R = self._conn.weights.shape[0]
        sfreq = self._cfg.temporal.sfreq
        T = self._cfg.temporal.n_samples_per_window
        window_ms = self._cfg.temporal.window_s * 1000.0
        dt_ms = self._cfg_tvb.integrator_dt_ms
        T_tvb = int(window_ms / dt_ms)

        seed_parcels = seed_parcel_ids(scenario, parc)
        temporal_pattern = build_stimulus_pattern(
            scenario,
            T=T_tvb,
            sfreq=1000.0 / dt_ms,
            amplitude=self._cfg_tvb.stimulus_amplitude,
            rng=rng,
        )
        weights_R = np.zeros(R, dtype=np.float64)
        if seed_parcels.size:
            weights_R[seed_parcels] = 1.0
        stim = self._make_stim(weights_R, temporal_pattern, dt_ms) if seed_parcels.size else None

        noise_seed = int(rng.integers(0, 2**31 - 1))
        warmup_ms = self._cfg_tvb.warmup_s * 1000.0
        total_ms = warmup_ms + window_ms

        baseline_raw = self._run_sim(noise_seed, total_ms, stim=None)
        stim_raw = self._run_sim(noise_seed, total_ms, stim=stim)

        keep = int(window_ms / dt_ms)
        baseline_raw = baseline_raw[-keep:]
        stim_raw = stim_raw[-keep:]

        baseline_R = _extract_observable(baseline_raw, self._cfg_tvb.model)
        stim_R = _extract_observable(stim_raw, self._cfg_tvb.model)

        sfreq_tvb = 1000.0 / dt_ms
        baseline_R = _resample_to(baseline_R, sfreq_tvb, sfreq, T)
        stim_R = _resample_to(stim_R, sfreq_tvb, sfreq, T)

        background = baseline_R[parc, :].astype(np.float32)
        source = np.zeros((N, T), dtype=np.float32)
        if seed_parcels.size:
            seed_mask = np.isin(parc, seed_parcels)
            diff_R = (stim_R - baseline_R).astype(np.float32)
            source[seed_mask] = diff_R[parc[seed_mask], :]
        return source, background

    def _run_sim(self, noise_seed: int, duration_ms: float, stim):
        sim = _build_simulator(self._cfg_tvb, self._tvb_conn, stim=stim)
        sim.integrator.noise.random_stream = np.random.RandomState(noise_seed)
        sim.configure()
        sim.simulation_length = duration_ms
        (raw_t, raw_s), = sim.run()
        return np.asarray(raw_s)
