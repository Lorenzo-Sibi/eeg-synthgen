from __future__ import annotations

from math import gcd
from typing import Any

import numpy as np

from synthgen.banks.connectivity import Connectivity
from synthgen.config import GenerationConfig
from synthgen.sample import Scenario, SourceSpace
from synthgen.sources.base import SourceGeneratorBackend
from synthgen.sources.tvb_stimulus import seed_parcel_ids

# Source-level preprocessing for the TVB backend follows the convention of
# DeepSIF (Sun et al. 2022, process_raw_nmm.m) and other TVB-based synthetic
# data papers: DC removal per region, no amplitude normalisation. The
# absolute output remains in TVB native mV-like units. Cross-backend
# amplitude alignment (TVB vs SEREEGA) is handled at sensor level by
# AcquisitionPipeline using a real-EEG-derived reference, not here.


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


def _build_jansen_rit(cfg_tvb, A_per_region: np.ndarray) -> Any:
    """DeepSIF-style Jansen-Rit with per-region excitability `A`."""
    from tvb.simulator import models

    return models.JansenRit(
        A=A_per_region.astype(np.float64),
        mu=np.array([cfg_tvb.jr_mu]),
        v0=np.array([cfg_tvb.jr_v0]),
        p_max=np.array([cfg_tvb.jr_p_max]),
        p_min=np.array([cfg_tvb.jr_p_min]),
    )


def _deepsif_noise_sigma(model, std: float) -> np.ndarray:
    """DeepSIF noise spec: sigma applied only to state-variable index 4.

    sigma[4] = (a * 3.25 * (p_max - p_min) * 0.5 * std) ** 2 / 2

    where `a` is the JR sigmoid steepness parameter (model.a), the 3.25
    factor matches the canonical excitatory PSP amplitude, and the entire
    expression yields the variance (not std) of the additive perturbation.
    Other state variables receive zero noise — matching DeepSIF exactly.
    """
    a = float(np.asarray(model.a).flat[0])
    p_max = float(np.asarray(model.p_max).flat[0])
    p_min = float(np.asarray(model.p_min).flat[0])
    phi_n_scaling = (a * 3.25 * (p_max - p_min) * 0.5 * float(std)) ** 2 / 2.0
    sigma = np.zeros(6, dtype=np.float64)
    sigma[4] = phi_n_scaling
    return sigma


def _build_simulator(cfg_tvb, tvb_conn, A_per_region: np.ndarray):
    """Build a DeepSIF-style TVB simulator: SigmoidalJansenRit coupling,
    per-region A on JansenRit model, HeunStochastic with targeted noise."""
    from tvb.simulator import coupling, integrators, monitors, noise, simulator

    model_name = cfg_tvb.model
    if model_name != "jansen_rit":
        # The DeepSIF-style elevation is JR-specific. For other models we
        # fall back to uniform A_baseline (no source elevation) and Linear
        # coupling, preserving the existing behaviour.
        from synthgen.sources.tvb_models import get_tvb_model

        model = get_tvb_model(model_name)
        cpl = coupling.Linear(a=np.array([cfg_tvb.global_coupling]))
        integ = integrators.HeunStochastic(
            dt=cfg_tvb.integrator_dt_ms,
            noise=noise.Additive(nsig=np.array([float(cfg_tvb.noise_std) ** 2])),
        )
    else:
        model = _build_jansen_rit(cfg_tvb, A_per_region)
        cpl = coupling.SigmoidalJansenRit(a=np.array([cfg_tvb.global_coupling]))
        sigma = _deepsif_noise_sigma(model, cfg_tvb.noise_std)
        integ = integrators.HeunStochastic(
            dt=cfg_tvb.integrator_dt_ms,
            noise=noise.Additive(nsig=sigma),
        )

    mon = monitors.TemporalAverage(period=cfg_tvb.integrator_dt_ms)
    sim = simulator.Simulator(
        model=model,
        connectivity=tvb_conn,
        coupling=cpl,
        integrator=integ,
        monitors=(mon,),
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
    """TVB-backed generator following the DeepSIF methodology (Sun et al. 2022,
    Rong et al. 2025).

    A *single* TVB simulation drives both source and background activity:
      - The Jansen-Rit excitability parameter ``A`` is elevated on the seed
        parcels (default 3.5 vs 3.25 baseline). The elevated regions become
        prone to spike-like activity under the network's noise and inputs.
      - The simulator output on seed parcels (broadcast to vertices via
        ``parcellation``) becomes ``source_activity``; non-seed vertices are
        zero (the framework's contract).
      - The same simulator output on *all* parcels (broadcast similarly)
        becomes ``background_activity``.

    No external ``StimuliRegion`` is injected. No paired with-vs-without
    subtraction is performed. This matches DeepSIF/generate_tvb_data.py.
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

        seed_parcels = seed_parcel_ids(scenario, parc)

        A_per_region = np.full(R, float(self._cfg_tvb.jr_A_baseline), dtype=np.float64)
        if seed_parcels.size:
            A_per_region[seed_parcels] = float(self._cfg_tvb.jr_A_seed)

        noise_seed = int(rng.integers(0, 2**31 - 1))
        warmup_ms = self._cfg_tvb.warmup_s * 1000.0
        total_ms = warmup_ms + window_ms

        raw = self._run_sim(noise_seed, total_ms, A_per_region)

        keep = int(window_ms / dt_ms)
        raw = raw[-keep:]
        y_R = _extract_observable(raw, self._cfg_tvb.model)  # (R, T_tvb)

        sfreq_tvb = 1000.0 / dt_ms
        y_R = _resample_to(y_R, sfreq_tvb, sfreq, T)         # (R, T)

        # DC removal per region (standard NMM preprocessing — same first step
        # as DeepSIF rescale_nmm_channel). Preserves native NMM amplitudes;
        # absolute-scale alignment with SEREEGA happens at sensor level.
        y_R = (y_R - y_R.mean(axis=1, keepdims=True)).astype(np.float32)

        background = y_R[parc, :].astype(np.float32)
        source = np.zeros((N, T), dtype=np.float32)
        if seed_parcels.size:
            seed_mask = np.isin(parc, seed_parcels)
            source[seed_mask] = y_R[parc[seed_mask], :]
        return source, background

    def _run_sim(
        self,
        noise_seed: int,
        duration_ms: float,
        A_per_region: np.ndarray,
    ) -> np.ndarray:
        sim = _build_simulator(self._cfg_tvb, self._tvb_conn, A_per_region)
        sim.integrator.noise.random_stream = np.random.RandomState(noise_seed)
        sim.configure()
        sim.simulation_length = duration_ms
        (raw_t, raw_s), = sim.run()
        return np.asarray(raw_s)
