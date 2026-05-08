import pytest
from pathlib import Path
from pydantic import ValidationError
from synthgen.config import (
    GenerationConfig,
    ScenarioPlanConfig,
    PriorFamilyWeights,
    NSourcesWeights,
    SEREEGABackendConfig,
    TemporalConfig,
)


def test_prior_family_weights_sum_to_one():
    w = PriorFamilyWeights()
    total = (
        w.local_contiguous
        + w.network_aware
        + w.state_dependent
        + w.broad_random
        + w.tvb_stub
    )
    assert abs(total - 1.0) < 1e-6


def test_n_sources_weights_sum_to_one():
    w = NSourcesWeights()
    assert abs(sum(w.weights.values()) - 1.0) < 1e-6


def test_temporal_config_defaults():
    t = TemporalConfig()
    assert t.sfreq == 500.0
    assert t.window_s == 1.0
    assert t.n_samples_per_window == 500
    assert t.signal_family_weights is None


def test_temporal_config_accepts_signal_family_weights():
    t = TemporalConfig(
        signal_families=["erp", "oscillatory_burst"],
        signal_family_weights=[0.25, 0.75],
    )
    assert t.signal_family_weights == [0.25, 0.75]


def test_temporal_config_rejects_bad_signal_family_weights():
    with pytest.raises(ValidationError):
        TemporalConfig(
            signal_families=["erp", "oscillatory_burst"],
            signal_family_weights=[1.0],
        )


def test_generation_config_from_yaml(tmp_path):
    yaml_content = """
anatomy_bank:
  bank_dir: banks/anatomy
leadfield_bank:
  bank_dir: banks/leadfield
montages:
  montages: []
writer:
  output_dir: data/generated
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = GenerationConfig.from_yaml(config_file)
    assert config.backend == "sereega"
    assert config.n_samples == 100_000
    assert config.global_seed == 42


def test_generation_config_rejects_unknown_backend(tmp_path):
    yaml_content = """
anatomy_bank:
  bank_dir: banks/anatomy
leadfield_bank:
  bank_dir: banks/leadfield
montages:
  montages: []
writer:
  output_dir: data/generated
backend: nonexistent_backend
"""
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(yaml_content)
    with pytest.raises(ValidationError):
        GenerationConfig.from_yaml(config_file)


def test_default_yaml_loads():
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    config = GenerationConfig.from_yaml(config_path)
    assert config.backend == "sereega"
    assert config.temporal.n_samples_per_window == 500
    core_montages = [m for m in config.montages.montages if m.split_role == "core"]
    assert len(core_montages) == 6
    prior_w = config.scenario_plan.prior_family_weights
    total = (
        prior_w.local_contiguous
        + prior_w.network_aware
        + prior_w.state_dependent
        + prior_w.broad_random
        + prior_w.tvb_stub
    )
    assert abs(total - 1.0) < 1e-6


def test_tvb_config_defaults_load():
    from synthgen.config import TVBBackendConfig
    c = TVBBackendConfig()
    assert c.model == "jansen_rit"
    assert c.warmup_s == 5.0
    assert c.reservoir_size == 100


def test_tvb_config_from_yaml(tmp_path):
    """Verify YAML loading wires through to TVBBackendConfig fields. Uses a
    fixture YAML so the test is independent of mutable config/default.yaml."""
    import yaml
    cfg_dict = {
        "anatomy_bank": {"bank_dir": "banks/anatomy"},
        "leadfield_bank": {"bank_dir": "banks/leadfield"},
        "montages": {"montages": []},
        "writer": {"output_dir": "out"},
        "tvb": {
            "model": "jansen_rit",
            "connectivity_scheme": "my_scheme",
            "reservoir_size": 42,
            "global_coupling": 0.05,
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg_dict))
    cfg = GenerationConfig.from_yaml(p)
    assert cfg.tvb.model == "jansen_rit"
    assert cfg.tvb.reservoir_size == 42
    assert cfg.tvb.connectivity_scheme == "my_scheme"
    assert cfg.tvb.global_coupling == pytest.approx(0.05)


def test_sereega_config_defaults_load():
    c = SEREEGABackendConfig()
    assert c.matlab_sereega_path is None
    assert c.erp_peak_count_weights == {1: 1.0}
    assert c.latency_jitter_s_range == (0.05, 0.30)
    assert c.patch_spatial_profile == "gaussian"


def test_sereega_config_from_yaml(tmp_path):
    import yaml

    cfg_dict = {
        "anatomy_bank": {"bank_dir": "banks/anatomy"},
        "leadfield_bank": {"bank_dir": "banks/leadfield"},
        "montages": {"montages": []},
        "writer": {"output_dir": "out"},
        "sereega": {
            "matlab_sereega_path": "external/SEREEGA",
            "erp_peak_count_weights": {1: 0.25, 2: 0.75},
            "patch_spatial_profile": "uniform",
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg_dict))
    cfg = GenerationConfig.from_yaml(p)
    assert cfg.sereega.matlab_sereega_path == Path("external/SEREEGA")
    assert cfg.sereega.erp_peak_count_weights == {1: 0.25, 2: 0.75}
    assert cfg.sereega.patch_spatial_profile == "uniform"


def test_noise_config_defaults():
    from synthgen.config import NoiseConfig

    c = NoiseConfig()
    assert c.snir_levels_db == [0.0, 5.0, 10.0, 15.0, 20.0]
    assert c.snr_sensor_levels_db == [0.0, 5.0, 10.0, 15.0, 20.0]


def test_noise_config_rejects_empty_levels():
    from synthgen.config import NoiseConfig

    with pytest.raises(ValidationError):
        NoiseConfig(snir_levels_db=[])
    with pytest.raises(ValidationError):
        NoiseConfig(snr_sensor_levels_db=[])


def test_sereega_config_rejects_invalid_range(tmp_path):
    yaml_content = """
anatomy_bank:
  bank_dir: banks/anatomy
leadfield_bank:
  bank_dir: banks/leadfield
montages:
  montages: []
writer:
  output_dir: out
sereega:
  erp_width_s_range: [0.08, 0.02]
"""
    p = tmp_path / "bad_sereega.yaml"
    p.write_text(yaml_content)
    with pytest.raises(ValidationError):
        GenerationConfig.from_yaml(p)
