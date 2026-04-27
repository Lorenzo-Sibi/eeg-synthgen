from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def _make_config(tmp_path):
    import yaml
    cfg = {
        "anatomy_bank": {"bank_dir": str(tmp_path / "anatomy"), "anatomy_ids": ["fsaverage"]},
        "leadfield_bank": {"bank_dir": str(tmp_path / "leadfield"), "conductivity_ids": ["standard"]},
        "montage_bank": {
            "bank_dir": str(tmp_path / "montage"),
            "montages": [
                {"name": "standard_1005_64", "n_channels": 64, "split_role": "core"},
                {"name": "standard_1005_21", "n_channels": 21, "split_role": "ood"},
            ],
        },
        "writer": {"output_dir": str(tmp_path / "out"), "chunk_size": 4},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    from synthgen.config import GenerationConfig
    return GenerationConfig.from_yaml(p)


# ── ScenarioSampler ──────────────────────────────────────────────────────────

def test_sampler_returns_scenario(tmp_path):
    from synthgen.scenario.sampler import ScenarioSampler
    from synthgen.sample import Scenario
    config = _make_config(tmp_path)
    sampler = ScenarioSampler(config)
    sc = sampler.sample(np.random.default_rng(0))
    assert isinstance(sc, Scenario)


def test_sampler_fields_populated(tmp_path):
    from synthgen.scenario.sampler import ScenarioSampler
    config = _make_config(tmp_path)
    sampler = ScenarioSampler(config)
    sc = sampler.sample(np.random.default_rng(1))
    assert sc.scenario_id != ""
    assert sc.anatomy_id in config.anatomy_bank.anatomy_ids
    assert sc.montage_id in [m.name for m in config.montage_bank.montages]
    assert sc.prior_family in ["local_contiguous", "network_aware", "state_dependent",
                                "broad_random", "tvb_stub"]
    assert sc.n_sources >= 1
    assert sc.difficulty in ["easy", "medium", "hard"]
    assert sc.signal_family in config.temporal.signal_families
    assert sc.split in ["train", "ood"]
    assert sc.leadfield_id == f"{sc.anatomy_id}__{sc.montage_id}__standard"


def test_sampler_ood_split_assigned_for_ood_montage(tmp_path):
    from synthgen.scenario.sampler import ScenarioSampler
    config = _make_config(tmp_path)
    sampler = ScenarioSampler(config)
    # sample many; verify that standard_1005_21 (ood) always gets split="ood"
    splits_by_montage: dict[str, set] = {}
    for i in range(200):
        sc = sampler.sample(np.random.default_rng(i))
        splits_by_montage.setdefault(sc.montage_id, set()).add(sc.split)
    if "standard_1005_21" in splits_by_montage:
        assert splits_by_montage["standard_1005_21"] == {"ood"}
    if "standard_1005_64" in splits_by_montage:
        assert splits_by_montage["standard_1005_64"] == {"train"}


def test_sampler_prior_family_distribution(tmp_path):
    from synthgen.scenario.sampler import ScenarioSampler
    config = _make_config(tmp_path)
    sampler = ScenarioSampler(config)
    rng = np.random.default_rng(42)
    families = [sampler.sample(rng).prior_family for _ in range(500)]
    counts = {f: families.count(f) for f in set(families)}
    # broad_random weight=0.15 → roughly 75/500; check it appears at all
    assert "broad_random" in counts
    assert "local_contiguous" in counts


def test_sampler_reproducible(tmp_path):
    from synthgen.scenario.sampler import ScenarioSampler
    config = _make_config(tmp_path)
    sampler = ScenarioSampler(config)
    sc1 = sampler.sample(np.random.default_rng(7))
    sc2 = sampler.sample(np.random.default_rng(7))
    assert sc1.scenario_id == sc2.scenario_id
    assert sc1.prior_family == sc2.prior_family
    assert sc1.n_sources == sc2.n_sources


def test_sampler_raises_if_no_core_montages(tmp_path):
    import yaml
    from synthgen.config import GenerationConfig
    from synthgen.scenario.sampler import ScenarioSampler
    cfg = {
        "anatomy_bank": {"bank_dir": str(tmp_path / "anatomy"), "anatomy_ids": ["fsaverage"]},
        "leadfield_bank": {"bank_dir": str(tmp_path / "leadfield"), "conductivity_ids": ["standard"]},
        "montage_bank": {
            "bank_dir": str(tmp_path / "montage"),
            "montages": [
                {"name": "standard_1005_21", "n_channels": 21, "split_role": "ood"},
            ],
        },
        "writer": {"output_dir": str(tmp_path / "out"), "chunk_size": 4},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    config = GenerationConfig.from_yaml(p)
    with pytest.raises(ValueError):
        ScenarioSampler(config)


# ── ZarrWriter ───────────────────────────────────────────────────────────────

import json
from dataclasses import asdict


def _make_eeg_sample(
    C: int = 8,
    T: int = 500,
    V: int = 20,
    anatomy_id: str = "fsaverage",
    montage_id: str = "standard_1005_64",
    rng=None,
):
    from synthgen.sample import EEGSample, Scenario
    if rng is None:
        rng = np.random.default_rng(0)
    sc = Scenario(
        scenario_id="s0000000000000000000",
        seed=0,
        anatomy_id=anatomy_id,
        leadfield_id=f"{anatomy_id}__{montage_id}__standard",
        montage_id=montage_id,
        reference_scheme="average",
        conductivity_id="standard",
        prior_family="broad_random",
        n_sources=1,
        signal_family="erp",
        difficulty="easy",
        split="train",
    )
    sc.snir_db = 10.0
    sc.snr_sensor_db = 15.0
    return EEGSample(
        eeg=rng.standard_normal((C, T)).astype(np.float32),
        source_activity=np.zeros((V, T), dtype=np.float32),
        source_support=np.zeros(V, dtype=bool),
        electrode_coords=np.zeros((C, 3), dtype=np.float32),
        source_coords=np.zeros((V, 3), dtype=np.float32),
        params=sc,
        snir_measured_db=10.0,
        snr_sensor_measured_db=15.0,
        active_area_cm2=3.0,
        config_hash="deadbeef",
    )


def test_zarr_writer_creates_store(tmp_path):
    from synthgen.writer.zarr_writer import ZarrWriter
    from synthgen.config import WriterConfig
    config = WriterConfig(output_dir=tmp_path / "out", chunk_size=4)
    writer = ZarrWriter(config)
    sample = _make_eeg_sample()
    writer.write(sample)
    writer.finalize()
    assert (tmp_path / "out" / "data.zarr").exists()
    assert (tmp_path / "out" / "metadata.jsonl").exists()


def test_zarr_writer_correct_shapes(tmp_path):
    import zarr
    from synthgen.writer.zarr_writer import ZarrWriter
    from synthgen.config import WriterConfig
    config = WriterConfig(output_dir=tmp_path / "out", chunk_size=4)
    writer = ZarrWriter(config)
    C, T, V = 8, 500, 20
    for i in range(3):
        writer.write(_make_eeg_sample(C=C, T=T, V=V, rng=np.random.default_rng(i)))
    writer.finalize()
    store = zarr.open(str(tmp_path / "out" / "data.zarr"), mode="r")
    grp = store["fsaverage__standard_1005_64"]
    assert grp["eeg"].shape == (3, C, T)
    assert grp["source_support"].shape == (3, V)
    assert grp["snir_db"].shape == (3,)
    assert grp["snr_sensor_db"].shape == (3,)
    assert grp["active_area_cm2"].shape == (3,)


def test_zarr_writer_metadata_jsonl(tmp_path):
    from synthgen.writer.zarr_writer import ZarrWriter
    from synthgen.config import WriterConfig
    config = WriterConfig(output_dir=tmp_path / "out", chunk_size=4)
    writer = ZarrWriter(config)
    for i in range(5):
        writer.write(_make_eeg_sample(rng=np.random.default_rng(i)))
    writer.finalize()
    lines = (tmp_path / "out" / "metadata.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5
    record = json.loads(lines[0])
    assert "scenario_id" in record
    assert "prior_family" in record


def test_zarr_writer_flush_on_chunk_boundary(tmp_path):
    import zarr
    from synthgen.writer.zarr_writer import ZarrWriter
    from synthgen.config import WriterConfig
    config = WriterConfig(output_dir=tmp_path / "out", chunk_size=2)
    writer = ZarrWriter(config)
    writer.write(_make_eeg_sample(rng=np.random.default_rng(0)))
    writer.write(_make_eeg_sample(rng=np.random.default_rng(1)))
    writer.finalize()
    store = zarr.open(str(tmp_path / "out" / "data.zarr"), mode="r")
    assert store["fsaverage__standard_1005_64"]["eeg"].shape[0] == 2


def test_zarr_writer_multi_group(tmp_path):
    import zarr
    from synthgen.writer.zarr_writer import ZarrWriter
    from synthgen.config import WriterConfig
    config = WriterConfig(output_dir=tmp_path / "out", chunk_size=4)
    writer = ZarrWriter(config)
    writer.write(_make_eeg_sample(C=64, montage_id="standard_1005_64"))
    writer.write(_make_eeg_sample(C=32, montage_id="standard_1005_32"))
    writer.finalize()
    store = zarr.open(str(tmp_path / "out" / "data.zarr"), mode="r")
    assert "fsaverage__standard_1005_64" in store
    assert "fsaverage__standard_1005_32" in store


def test_zarr_writer_manifest_in_attrs(tmp_path):
    import zarr
    from synthgen.writer.zarr_writer import ZarrWriter
    from synthgen.config import WriterConfig
    config = WriterConfig(output_dir=tmp_path / "out", chunk_size=4)
    writer = ZarrWriter(config)
    writer.write(_make_eeg_sample())
    writer.finalize()
    store = zarr.open(str(tmp_path / "out" / "data.zarr"), mode="r")
    assert "n_samples" in store.attrs
    assert store.attrs["n_samples"] == 1


# ── PipelineRunner ───────────────────────────────────────────────────────────

def _make_bank_files(tmp_path, N: int = 20, C: int = 8, scheme: str = "desikan_killiany"):
    """Create minimal anatomy, leadfield, and montage bank files."""
    anatomy_dir = tmp_path / "anatomy" / "fsaverage"
    (anatomy_dir / "parcellations").mkdir(parents=True)
    adj = sp.eye(N, format="csr", dtype=np.float32)
    np.savez_compressed(
        anatomy_dir / "source_space.npz",
        vertex_coords=np.zeros((N, 3), dtype=np.float32),
        adjacency_data=adj.data.astype(np.float32),
        adjacency_indices=adj.indices.astype(np.int32),
        adjacency_indptr=adj.indptr.astype(np.int32),
        adjacency_shape=np.array([N, N], dtype=np.int32),
        hemisphere=np.zeros(N, dtype=np.int32),
    )
    np.savez_compressed(
        anatomy_dir / "parcellations" / f"{scheme}.npz",
        parcellation=np.zeros(N, dtype=np.int32),
        region_labels=np.array(["r0"], dtype=str),
        scheme=scheme,
    )
    lf_dir = tmp_path / "leadfield" / "fsaverage" / "standard_1005_64"
    lf_dir.mkdir(parents=True)
    G = np.random.default_rng(0).standard_normal((C, N)).astype(np.float32) * 0.01
    np.savez_compressed(lf_dir / "standard.npz", G=G)
    montage_dir = tmp_path / "montage"
    montage_dir.mkdir(parents=True)
    ch_names = np.array([f"EEG{i:03d}" for i in range(C)], dtype=str)
    np.savez_compressed(
        montage_dir / "standard_1005_64.npz",
        coords=np.zeros((C, 3), dtype=np.float32),
        ch_names=ch_names,
    )


def _make_runner_config(tmp_path, N: int = 20, C: int = 8, n_samples: int = 3):
    import yaml
    cfg = {
        "anatomy_bank": {"bank_dir": str(tmp_path / "anatomy"), "anatomy_ids": ["fsaverage"]},
        "leadfield_bank": {"bank_dir": str(tmp_path / "leadfield"), "conductivity_ids": ["standard"]},
        "montage_bank": {
            "bank_dir": str(tmp_path / "montage"),
            "montages": [{"name": "standard_1005_64", "n_channels": C, "split_role": "core"}],
        },
        "writer": {"output_dir": str(tmp_path / "out"), "chunk_size": 256},
        "qc": {"min_valid_channels": C},
        "n_samples": n_samples,
        "n_workers": 1,
        "global_seed": 42,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    from synthgen.config import GenerationConfig
    return GenerationConfig.from_yaml(p)


def test_pipeline_runner_generates_n_samples(tmp_path):
    import zarr
    from synthgen.pipeline_runner import PipelineRunner
    _make_bank_files(tmp_path)
    config = _make_runner_config(tmp_path, n_samples=3)
    runner = PipelineRunner(config)
    runner.run()
    store = zarr.open(str(tmp_path / "out" / "data.zarr"), mode="r")
    total = sum(store[k]["eeg"].shape[0] for k in store.group_keys())
    assert total == 3


def test_pipeline_runner_metadata_jsonl(tmp_path):
    from synthgen.pipeline_runner import PipelineRunner
    _make_bank_files(tmp_path)
    config = _make_runner_config(tmp_path, n_samples=4)
    runner = PipelineRunner(config)
    runner.run()
    lines = (tmp_path / "out" / "metadata.jsonl").read_text().strip().split("\n")
    assert len(lines) == 4


def test_pipeline_runner_reproducible(tmp_path):
    import zarr
    from synthgen.pipeline_runner import PipelineRunner
    _make_bank_files(tmp_path)

    def _run(out_dir):
        import yaml
        raw = yaml.safe_load(open(tmp_path / "cfg.yaml"))
        raw["writer"]["output_dir"] = str(out_dir)
        p2 = tmp_path / "cfg2.yaml"
        p2.write_text(yaml.dump(raw))
        from synthgen.config import GenerationConfig
        c2 = GenerationConfig.from_yaml(p2)
        PipelineRunner(c2).run()
        store = zarr.open(str(out_dir / "data.zarr"), mode="r")
        key = list(store.group_keys())[0]
        return store[key]["eeg"][:]

    # Need a config file to exist for _run to read; create it first
    _make_runner_config(tmp_path, n_samples=2)

    eeg1 = _run(tmp_path / "run1")
    eeg2 = _run(tmp_path / "run2")
    np.testing.assert_array_equal(eeg1, eeg2)


# ── validate_dataset script ───────────────────────────────────────────────────

def test_validate_dataset_passes_on_valid_store(tmp_path):
    from synthgen.pipeline_runner import PipelineRunner
    from scripts.validate_dataset import validate
    _make_bank_files(tmp_path)
    config = _make_runner_config(tmp_path, n_samples=2)
    PipelineRunner(config).run()
    # validate should not raise
    validate(tmp_path / "out")


def test_validate_dataset_fails_on_missing_store(tmp_path):
    from scripts.validate_dataset import validate
    with pytest.raises(AssertionError):
        validate(tmp_path / "nonexistent")


def test_pipeline_selects_tvb_backend_when_configured(tmp_path):
    pytest.importorskip("tvb")
    from synthgen.config import (
        AnatomyBankConfig, ConnectivityBankConfig, GenerationConfig,
        LeadfieldBankConfig, MontageBankConfig, MontageEntry,
        TVBBackendConfig, WriterConfig,
    )
    from synthgen.sources.tvb_backend import TVBSourceGenerator
    from synthgen.pipeline_runner import PipelineRunner

    conn_dir = tmp_path / "c"; conn_dir.mkdir()
    np.savez_compressed(
        conn_dir / "desikan_killiany.npz",
        weights=np.zeros((4, 4), dtype=np.float32),
        tract_lengths=np.ones((4, 4), dtype=np.float32) * 10.0,
        region_centers=np.zeros((4, 3), dtype=np.float32),
        region_labels=np.array(["a", "b", "c", "d"], dtype=str),
        scheme="desikan_killiany",
    )
    cfg = GenerationConfig(
        anatomy_bank=AnatomyBankConfig(bank_dir=tmp_path / "a"),
        leadfield_bank=LeadfieldBankConfig(bank_dir=tmp_path / "l"),
        montage_bank=MontageBankConfig(
            bank_dir=tmp_path / "m",
            montages=[MontageEntry(name="standard_1005_64", n_channels=64, split_role="core")],
        ),
        connectivity_bank=ConnectivityBankConfig(bank_dir=conn_dir),
        writer=WriterConfig(output_dir=tmp_path / "out"),
        backend="tvb",
        tvb=TVBBackendConfig(warmup_s=0.2, reservoir_duration_s=1.0, reservoir_size=4),
    )
    runner = PipelineRunner(cfg)
    assert isinstance(runner._backend, TVBSourceGenerator)
