import pytest
from pathlib import Path
from pydantic import ValidationError
from synthgen.config import (
    GenerationConfig,
    ScenarioPlanConfig,
    PriorFamilyWeights,
    NSourcesWeights,
    DifficultyWeights,
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


def test_difficulty_weights_sum_to_one():
    w = DifficultyWeights()
    assert abs(w.easy + w.medium + w.hard - 1.0) < 1e-6


def test_temporal_config_defaults():
    t = TemporalConfig()
    assert t.sfreq == 500.0
    assert t.window_s == 1.0
    assert t.n_samples_per_window == 500


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
