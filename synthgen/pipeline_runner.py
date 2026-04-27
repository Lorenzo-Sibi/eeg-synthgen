from __future__ import annotations

import numpy as np

from synthgen.acquisition.pipeline import AcquisitionPipeline
from synthgen.banks.anatomy import AnatomyBank
from synthgen.banks.connectivity import ConnectivityBank
from synthgen.banks.leadfield import LeadfieldBank
from synthgen.banks.montage import MontageBank
from synthgen.config import GenerationConfig
from synthgen.scenario.sampler import ScenarioSampler
from synthgen.sources.priors.broad_random import BroadRandomPrior
from synthgen.sources.priors.local_contiguous import LocalContiguousPrior
from synthgen.sources.priors.network_aware import NetworkAwarePrior
from synthgen.sources.priors.state_dependent import StateDependentPrior
from synthgen.sources.sereega_backend import SEREEGABackend
from synthgen.sources.tvb_backend import TVBSourceGenerator
from synthgen.writer.zarr_writer import ZarrWriter

MAX_ATTEMPTS_MULTIPLIER = 100

_PRIOR_REGISTRY: dict[str, type] = {
    "local_contiguous": LocalContiguousPrior,
    "network_aware": NetworkAwarePrior,
    "state_dependent": StateDependentPrior,
    "broad_random": BroadRandomPrior,
    "tvb_stub": BroadRandomPrior,  # TVB backend not yet implemented
}


class PipelineRunner:
    """Sequential generation loop: sample -> prior -> backend -> acquisition -> write.

    Runs until ``config.n_samples`` QC-passing samples have been written.
    Bank files that are not found on disk cause the scenario to be skipped
    (useful when only a subset of banks has been prepared).
    """

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config
        self._anatomy_bank = AnatomyBank(config.anatomy_bank)
        self._leadfield_bank = LeadfieldBank(config.leadfield_bank)
        self._montage_bank = MontageBank(config.montage_bank)
        if config.backend == "sereega":
            self._backend = SEREEGABackend(config)
        elif config.backend == "tvb":
            self._connectivity_bank = ConnectivityBank(config.connectivity_bank)
            conn = self._connectivity_bank.load(config.tvb.connectivity_scheme)
            self._backend = TVBSourceGenerator(config, conn)
        else:
            raise ValueError(f"Unknown backend {config.backend!r}")
        self._pipeline = AcquisitionPipeline(config)
        self._sampler = ScenarioSampler(config)
        self._writer = ZarrWriter(config.writer)

    def run(self) -> None:
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
                scenario_rng = np.random.default_rng(scenario.seed)

                # Load anatomy, leadfield, montage - skip if bank not prepared
                try:
                    source_space = self._anatomy_bank.load(scenario.anatomy_id)
                    leadfield = self._leadfield_bank.load(scenario.leadfield_id)
                    montage = self._montage_bank.load(scenario.montage_id)
                except FileNotFoundError:
                    n_skipped_bank += 1
                    continue

                # Fill spatial prior fields
                prior = _PRIOR_REGISTRY[scenario.prior_family]()
                scenario = prior.sample(scenario, source_space, scenario_rng)

                # Generate source-space signals
                source_activity, background_activity = self._backend.generate(
                    scenario, source_space, scenario_rng
                )

                # Acquisition: project → noise → artifact → reference → QC
                sample, qc = self._pipeline.run(
                    scenario,
                    source_space,
                    source_activity,
                    background_activity,
                    leadfield,
                    montage.coords,
                    montage.ch_names,
                    scenario_rng,
                )

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
