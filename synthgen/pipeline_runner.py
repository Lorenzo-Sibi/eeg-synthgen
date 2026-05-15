from __future__ import annotations

import multiprocessing

import numpy as np
import tqdm

from synthgen.acquisition.pipeline import AcquisitionPipeline
from synthgen.banks.anatomy import AnatomyBank
from synthgen.banks.connectivity import ConnectivityBank
from synthgen.banks.leadfield import LeadfieldBank
from synthgen.config import GenerationConfig
from synthgen.sample import Scenario
from synthgen.scenario.sampler import ScenarioSampler
from synthgen.sources.priors.broad_random import BroadRandomPrior
from synthgen.sources.priors.local_contiguous import LocalContiguousPrior
from synthgen.sources.priors.network_aware import NetworkAwarePrior
from synthgen.sources.priors.state_dependent import StateDependentPrior
from synthgen.sources.sereega_backend import SEREEGABackend
from synthgen.sources.tvb_backend import TVBSourceGenerator
from synthgen.writer.zarr_writer import ZarrWriter

MAX_ATTEMPTS_MULTIPLIER = 100
MAX_WORKERS = 4  # Remember, every process it's a MATLAB instace, so change this accordingly. Please, be nice to your ram ;)

_PRIOR_REGISTRY: dict[str, type] = {
    "local_contiguous": LocalContiguousPrior,
    "network_aware": NetworkAwarePrior,
    "state_dependent": StateDependentPrior,
    "broad_random": BroadRandomPrior,
    "tvb_stub": BroadRandomPrior,  # TVB backend not yet implemented
}


_WORKER_RUNNER: "PipelineRunner | None" = None


def _init_worker(config_dict: dict) -> None:
    global _WORKER_RUNNER
    print(f"Worker {multiprocessing.current_process().name} initializing...", flush=True)
    _WORKER_RUNNER = PipelineRunner(
        GenerationConfig.model_validate(config_dict),
        load_runtime=True,
    )


def _produce_one_in_worker(scenario: Scenario):
    if _WORKER_RUNNER is None:
        raise RuntimeError("Worker runner was not initialized")
    return _WORKER_RUNNER._worker_func(scenario)


class PipelineRunner:
    """Generation loop: sample -> prior -> backend -> acquisition -> write.

    Runs until ``config.n_samples`` QC-passing samples have been written.
    Bank files that are not found on disk cause the scenario to be skipped
    (useful when only a subset of banks has been prepared).
    """

    def __init__(self, config: GenerationConfig, *, load_runtime: bool | None = None) -> None:
        self._config = config
        self._anatomy_bank = None
        self._leadfield_bank = None
        self._connectivity_bank = None
        self._backend = None
        self._pipeline = None
        self._sampler = ScenarioSampler(config)
        self._writer = ZarrWriter(config.writer)

        if load_runtime is None:
            load_runtime = config.n_workers <= 1 or config.backend == "tvb"
        if load_runtime:
            self._init_runtime()

    def _init_runtime(self) -> None:
        if self._backend is not None:
            return
        config = self._config
        self._anatomy_bank = AnatomyBank(config.anatomy_bank)
        self._leadfield_bank = LeadfieldBank(config.leadfield_bank)
        self._tvb_backend = None
        if config.backend == "sereega":
            self._backend = SEREEGABackend(config)
        elif config.backend == "tvb":
            self._connectivity_bank = ConnectivityBank(config.connectivity_bank)
            conn = self._connectivity_bank.load(config.tvb.connectivity_scheme)
            self._backend = TVBSourceGenerator(config, conn)
        elif config.backend == "mix":
            self._backend = SEREEGABackend(config)
            self._connectivity_bank = ConnectivityBank(config.connectivity_bank)
            conn = self._connectivity_bank.load(config.tvb.connectivity_scheme)
            self._tvb_backend = TVBSourceGenerator(config, conn)
        else:
            raise ValueError(f"Unknown backend {config.backend!r}")
        self._pipeline = AcquisitionPipeline(config)

    def _worker_func(self, scenario: Scenario):
        self._init_runtime()
        assert self._anatomy_bank is not None
        assert self._leadfield_bank is not None
        assert self._backend is not None
        assert self._pipeline is not None

        scenario_rng = np.random.default_rng(scenario.seed)

        # Load anatomy and leadfield bundle - skip if bank not prepared
        try:
            source_space = self._anatomy_bank.load(scenario.anatomy_id)
            leadfield = self._leadfield_bank.load(scenario.leadfield_id)
        except FileNotFoundError:
            return (None, None, scenario)

        # Fill spatial prior fields
        prior = _PRIOR_REGISTRY[scenario.prior_family]()
        scenario = prior.sample(scenario, source_space, scenario_rng)

        # Generate source-space signals; for backend="mix" route per-scenario
        # via the scenario RNG (deterministic from scenario.seed).
        if self._tvb_backend is not None and scenario_rng.random() < self._config.mix_tvb_fraction:
            scenario.backend_used = "tvb"
            source_activity, background_activity = self._tvb_backend.generate(
                scenario, source_space, scenario_rng
            )
        else:
            scenario.backend_used = "sereega" if self._config.backend != "tvb" else "tvb"
            source_activity, background_activity = self._backend.generate(
                scenario, source_space, scenario_rng
            )

        # Acquisition: project -> noise -> artifact -> reference -> QC
        sample, qc = self._pipeline.run(
            scenario,
            source_space,
            source_activity,
            background_activity,
            leadfield.G,
            leadfield.electrode_coords,
            leadfield.ch_names,
            scenario_rng,
        )
        return (sample, qc, scenario)

    def run(self) -> None:
        if self._config.n_workers <= 1 or self._config.backend in ("tvb", "mix"):
            return self._run_sequential()
        return self._run_parallel()

    def _run_parallel(self) -> None:
        if self._config.n_workers > MAX_WORKERS:
            raise ValueError(f"n_workers={self._config.n_workers} exceeds maximum of {MAX_WORKERS}")

        with self._writer:
            rng = np.random.default_rng(self._config.global_seed)
            n_target = self._config.n_samples
            n_written = 0
            n_skipped_bank = 0
            n_skipped_qc = 0

            max_attempts = n_target * MAX_ATTEMPTS_MULTIPLIER
            n_attempts = 0

            with multiprocessing.Pool(
                self._config.n_workers,
                initializer=_init_worker,
                initargs=(self._config.model_dump(mode="python"),),
            ) as pool:
                scenario_iter = (self._sampler.sample(rng) for _ in range(max_attempts))
                results_iter = pool.imap_unordered(
                    _produce_one_in_worker,
                    scenario_iter,
                    chunksize=1,
                )
                
                for sample, qc, scenario in tqdm.tqdm(results_iter):
                    n_attempts += 1

                    if sample is None or qc is None or scenario is None:
                        n_skipped_bank += 1
                    elif not qc.passed:
                        n_skipped_qc += 1
                    else:
                        self._writer.write(sample)
                        n_written += 1

                    if n_written >= n_target:
                        break

            if n_written < n_target:
                raise RuntimeError(
                    f"Exceeded {max_attempts} attempts to generate {n_target} samples "
                    f"(written={n_written}, skipped_bank={n_skipped_bank}, "
                    f"skipped_qc={n_skipped_qc}, completed={n_attempts}). "
                    "Check bank paths and QC thresholds."
                )

        print(
            f"Done. Written={n_written}, "
            f"skipped_bank={n_skipped_bank}, skipped_qc={n_skipped_qc}"
        )

    def _run_sequential(self) -> None:
        with self._writer:
            rng = np.random.default_rng(self._config.global_seed)
            n_target = self._config.n_samples
            n_written = 0
            n_skipped_bank = 0
            n_skipped_qc = 0

            max_attempts = n_target * MAX_ATTEMPTS_MULTIPLIER
            n_attempts = 0

            while n_written < n_target:
                if n_attempts >= max_attempts:
                    raise RuntimeError(
                        f"Exceeded {max_attempts} attempts to generate {n_target} samples "
                        f"(written={n_written}, skipped_bank={n_skipped_bank}, "
                        f"skipped_qc={n_skipped_qc}). Check bank paths and QC thresholds."
                    )
                n_attempts += 1

                scenario = self._sampler.sample(rng)
                sample, qc, scenario = self._worker_func(scenario=scenario)

                if sample is None or qc is None or scenario is None:
                    n_skipped_bank += 1
                    continue

                if not qc.passed:
                    n_skipped_qc += 1
                    continue

                self._writer.write(sample)
                n_written += 1

                if n_written % 100 == 0:
                    print(
                        f"  [{n_written}/{n_target}] "
                        f"skipped bank={n_skipped_bank} qc={n_skipped_qc}"
                    )

            # finalize() called automatically by __exit__
        print(
            f"Done. Written={n_written}, "
            f"skipped_bank={n_skipped_bank}, skipped_qc={n_skipped_qc}"
        )
