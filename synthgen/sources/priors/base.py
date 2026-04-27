from abc import ABC, abstractmethod

import numpy as np

from synthgen.sample import Scenario, SourceSpace


class ActivationPrior(ABC):
    @abstractmethod
    def sample(
        self,
        scenario: Scenario,
        source_space: SourceSpace,
        rng: np.random.Generator,
    ) -> Scenario:
        """Fill spatial parameters on Scenario and return it."""