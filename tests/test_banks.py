import numpy as np
import pytest
import scipy.sparse as sp

from synthgen.config import AnatomyBankConfig, LeadfieldBankConfig
from synthgen.banks.anatomy import AnatomyBank
from synthgen.sample import SourceSpace


def _make_source_space_npz(
    tmp_path, anatomy_id: str, N: int = 50, scheme: str = "desikan_killiany"
) -> None:
    adj = sp.eye(N, format="csr", dtype=np.float32)
    d = tmp_path / "anatomy" / anatomy_id
    (d / "parcellations").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "source_space.npz",
        vertex_coords=np.zeros((N, 3), dtype=np.float32),
        adjacency_data=adj.data,
        adjacency_indices=adj.indices.astype(np.int32),
        adjacency_indptr=adj.indptr.astype(np.int32),
        adjacency_shape=np.array([N, N], dtype=np.int32),
        hemisphere=np.zeros(N, dtype=np.int32),
    )
    np.savez_compressed(
        d / "parcellations" / f"{scheme}.npz",
        parcellation=np.zeros(N, dtype=np.int32),
        region_labels=np.array(["r0"], dtype=str),
        scheme=scheme,
    )


def test_anatomy_bank_loads_source_space(tmp_path):
    _make_source_space_npz(tmp_path, "fsaverage", N=50)
    config = AnatomyBankConfig(bank_dir=tmp_path / "anatomy")
    bank = AnatomyBank(config)
    ss = bank.load("fsaverage")
    assert isinstance(ss, SourceSpace)
    assert ss.vertex_coords.shape == (50, 3)
    assert ss.parcellation.shape == (50,)
    assert ss.hemisphere.shape == (50,)
    assert ss.adjacency.shape == (50, 50)


def test_anatomy_bank_caches_in_memory(tmp_path):
    _make_source_space_npz(tmp_path, "fsaverage", N=50)
    config = AnatomyBankConfig(bank_dir=tmp_path / "anatomy")
    bank = AnatomyBank(config)
    ss1 = bank.load("fsaverage")
    ss2 = bank.load("fsaverage")
    assert ss1 is ss2


def test_anatomy_bank_raises_for_missing(tmp_path):
    config = AnatomyBankConfig(bank_dir=tmp_path / "anatomy")
    bank = AnatomyBank(config)
    with pytest.raises(FileNotFoundError):
        bank.load("nonexistent")


from synthgen.banks.leadfield import LeadfieldBank, LeadfieldData


def _make_leadfield_npz(tmp_path, anatomy_id: str, montage_id: str,
                         conductivity_id: str, C: int = 64, N: int = 50) -> None:
    d = tmp_path / "leadfield" / anatomy_id / montage_id
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"{conductivity_id}.npz",
        G=np.random.randn(C, N).astype(np.float32),
        ch_names=np.array([f"Ch{i:03d}" for i in range(C)]),
        electrode_coords=np.random.randn(C, 3).astype(np.float32),
    )


def test_leadfield_bank_loads_bundle(tmp_path):
    _make_leadfield_npz(tmp_path, "fsaverage", "standard_1005_64", "standard", C=64, N=50)
    config = LeadfieldBankConfig(bank_dir=tmp_path / "leadfield")
    bank = LeadfieldBank(config)
    lf = bank.load("fsaverage__standard_1005_64__standard")
    assert isinstance(lf, LeadfieldData)
    assert lf.G.shape == (64, 50)
    assert lf.G.dtype == np.float32
    assert lf.ch_names == [f"Ch{i:03d}" for i in range(64)]
    assert lf.electrode_coords.shape == (64, 3)
    assert lf.electrode_coords.dtype == np.float32


def test_leadfield_bank_caches_in_memory(tmp_path):
    _make_leadfield_npz(tmp_path, "fsaverage", "standard_1005_64", "standard")
    config = LeadfieldBankConfig(bank_dir=tmp_path / "leadfield")
    bank = LeadfieldBank(config)
    lf1 = bank.load("fsaverage__standard_1005_64__standard")
    lf2 = bank.load("fsaverage__standard_1005_64__standard")
    assert lf1 is lf2


def test_leadfield_bank_raises_for_missing(tmp_path):
    config = LeadfieldBankConfig(bank_dir=tmp_path / "leadfield")
    bank = LeadfieldBank(config)
    with pytest.raises(FileNotFoundError):
        bank.load("fsaverage__standard_1005_64__standard")


def test_leadfield_bank_raises_for_invalid_id(tmp_path):
    config = LeadfieldBankConfig(bank_dir=tmp_path / "leadfield")
    bank = LeadfieldBank(config)
    with pytest.raises(ValueError, match="anatomy__montage__conductivity"):
        bank.load("bad_id_format")


