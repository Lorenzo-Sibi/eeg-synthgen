from abc import ABC, abstractmethod

import numpy as np

from synthgen.sample import Scenario, SourceSpace


class SourceGeneratorBackend(ABC):
    @abstractmethod
    def generate(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (source_activity: NxT, background_activity: NxT)."""
