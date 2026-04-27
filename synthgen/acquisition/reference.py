from abc import ABC, abstractmethod

import numpy as np


class ReferenceOperator(ABC):
    @abstractmethod
    def apply(self, eeg: np.ndarray) -> np.ndarray:
        """Apply reference scheme to eeg (CxT). Returns re-referenced eeg: CxT."""


class AverageReference(ReferenceOperator):
    def apply(self, eeg: np.ndarray) -> np.ndarray:
        return eeg - eeg.mean(axis=0, keepdims=True)


class FixedReference(ReferenceOperator):
    def __init__(self, channel_index: int) -> None:
        self.channel_index = channel_index

    def apply(self, eeg: np.ndarray) -> np.ndarray:
        return eeg - eeg[self.channel_index, :]


class NoReference(ReferenceOperator):
    def apply(self, eeg: np.ndarray) -> np.ndarray:
        return eeg