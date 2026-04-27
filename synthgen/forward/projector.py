from abc import ABC, abstractmethod

import numpy as np


class ForwardProjector(ABC):
    @abstractmethod
    def project(self, source_activity: np.ndarray, leadfield: np.ndarray) -> np.ndarray:
        """Apply leadfield to source_activity. Returns clean_eeg: CxT.
        Parameters:
        - source_activity: NxT ndarray of source activations
        - leadfield CxN ndarray mapping source space to sensor space (on the scalp)
        """


class LinearProjector(ForwardProjector):
    def project(self, source_activity: np.ndarray, leadfield: np.ndarray) -> np.ndarray:
        return leadfield @ source_activity