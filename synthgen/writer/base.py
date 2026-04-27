from abc import ABC, abstractmethod

from synthgen.sample import EEGSample


class DatasetWriter(ABC):
    @abstractmethod
    def write(self, sample: EEGSample) -> None:
        """Persist one EEGSample to the underlying store."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush buffers, close handles, and write manifest/provenance."""
