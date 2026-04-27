import numpy as np
import pytest

from synthgen.acquisition.artifacts import ArtifactEngine
from synthgen.acquisition.noise import SensorNoiseEngine
from synthgen.acquisition.reference import AverageReference, FixedReference, NoReference, ReferenceOperator
from synthgen.forward.projector import ForwardProjector, LinearProjector
from synthgen.sources.base import SourceGeneratorBackend
from synthgen.sources.priors.base import ActivationPrior
from synthgen.sources.tvb_backend import TVBSourceGenerator
from synthgen.writer.base import DatasetWriter


def test_source_generator_backend_is_abstract():
    with pytest.raises(TypeError):
        SourceGeneratorBackend()


def test_activation_prior_is_abstract():
    with pytest.raises(TypeError):
        ActivationPrior()


def test_forward_projector_is_abstract():
    with pytest.raises(TypeError):
        ForwardProjector()


def test_sensor_noise_engine_is_abstract():
    with pytest.raises(TypeError):
        SensorNoiseEngine()


def test_artifact_engine_is_abstract():
    with pytest.raises(TypeError):
        ArtifactEngine()


def test_dataset_writer_is_abstract():
    with pytest.raises(TypeError):
        DatasetWriter()


def test_concrete_backend_satisfies_interface():
    class _Stub(SourceGeneratorBackend):
        def generate(self, scenario, source_space, rng):
            raise NotImplementedError

    stub = _Stub()
    assert isinstance(stub, SourceGeneratorBackend)


def test_concrete_prior_satisfies_interface():
    class _Stub(ActivationPrior):
        def sample(self, scenario, source_space, rng):
            raise NotImplementedError

    stub = _Stub()
    assert isinstance(stub, ActivationPrior)


def test_concrete_projector_satisfies_interface():
    class _Stub(ForwardProjector):
        def project(self, source_activity, leadfield):
            raise NotImplementedError

    stub = _Stub()
    assert isinstance(stub, ForwardProjector)


def test_concrete_writer_satisfies_interface():
    class _Stub(DatasetWriter):
        def write(self, sample):
            raise NotImplementedError

        def finalize(self):
            raise NotImplementedError

    stub = _Stub()
    assert isinstance(stub, DatasetWriter)


def test_reference_operator_is_abstract():
    with pytest.raises(TypeError):
        ReferenceOperator()


def test_average_reference_subtracts_mean():
    eeg = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = AverageReference().apply(eeg)
    assert np.allclose(result.mean(axis=0), 0.0)


def test_fixed_reference_subtracts_channel():
    eeg = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = FixedReference(channel_index=0).apply(eeg)
    assert np.allclose(result[0], 0.0)


def test_no_reference_is_identity():
    eeg = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = NoReference().apply(eeg)
    assert np.allclose(result, eeg)


def test_linear_projector_shape():
    C, N, T = 64, 100, 500
    leadfield = np.random.randn(C, N)
    sources = np.random.randn(N, T)
    eeg = LinearProjector().project(sources, leadfield)
    assert eeg.shape == (C, T)


def test_tvb_backend_importable():
    assert TVBSourceGenerator.__name__ == "TVBSourceGenerator"
