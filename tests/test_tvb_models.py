import pytest

pytest.importorskip("tvb")

from synthgen.sources.tvb_models import get_tvb_model, model_output_channel


def test_jansen_rit_returns_model():
    m = get_tvb_model("jansen_rit")
    assert m.__class__.__name__ == "JansenRit"


def test_generic_2d_returns_model():
    m = get_tvb_model("generic_2d_oscillator")
    assert m.__class__.__name__ == "Generic2dOscillator"


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_tvb_model("nope")


def test_output_channel_known():
    assert model_output_channel("jansen_rit") == 1


def test_output_channel_unknown_raises():
    with pytest.raises(ValueError):
        model_output_channel("nope")