import numpy as np
import pytest
import scipy.sparse as sp
from synthgen.banks.anatomy import AnatomyBank
from synthgen.config import AnatomyBankConfig


def _write_geometry(anat_dir, N=10):
    A = sp.eye(N, format="csr", dtype=np.float32)
    np.savez_compressed(
        anat_dir / "source_space.npz",
        vertex_coords=np.zeros((N, 3), dtype=np.float32),
        adjacency_data=A.data,
        adjacency_indices=A.indices,
        adjacency_indptr=A.indptr,
        adjacency_shape=np.array(A.shape, dtype=np.int32),
        hemisphere=np.zeros(N, dtype=np.int32),
    )


def _write_parcellation(parc_dir, scheme, parcellation, labels):
    np.savez_compressed(
        parc_dir / f"{scheme}.npz",
        parcellation=parcellation.astype(np.int32),
        region_labels=np.array(labels, dtype=str),
        scheme=scheme,
    )


def test_anatomy_bank_loads_default_scheme(tmp_path):
    anat_dir = tmp_path / "dk"
    (anat_dir / "parcellations").mkdir(parents=True)
    _write_geometry(anat_dir, N=10)
    _write_parcellation(
        anat_dir / "parcellations", "desikan_killiany",
        np.zeros(10, dtype=np.int32), ["bankssts-lh", "caudal-lh"],
    )
    bank = AnatomyBank(AnatomyBankConfig(
        bank_dir=tmp_path, parcellation_scheme="desikan_killiany",
    ))
    ss = bank.load("dk")
    assert ss.parcellation_scheme == "desikan_killiany"
    assert ss.region_labels == ["bankssts-lh", "caudal-lh"]


def test_anatomy_bank_load_with_scheme_override(tmp_path):
    anat_dir = tmp_path / "multi"
    (anat_dir / "parcellations").mkdir(parents=True)
    _write_geometry(anat_dir, N=10)
    _write_parcellation(
        anat_dir / "parcellations", "desikan_killiany",
        np.zeros(10, dtype=np.int32), ["a", "b"],
    )
    _write_parcellation(
        anat_dir / "parcellations", "destrieux",
        (np.arange(10) % 3).astype(np.int32), ["x", "y", "z"],
    )
    bank = AnatomyBank(AnatomyBankConfig(
        bank_dir=tmp_path, parcellation_scheme="desikan_killiany",
    ))
    assert bank.load("multi").parcellation_scheme == "desikan_killiany"
    assert bank.load("multi", scheme="destrieux").parcellation_scheme == "destrieux"
    assert len(bank.load("multi", scheme="destrieux").region_labels) == 3


def test_anatomy_bank_missing_geometry_raises(tmp_path):
    bank = AnatomyBank(AnatomyBankConfig(bank_dir=tmp_path))
    with pytest.raises(FileNotFoundError, match="Source space not found"):
        bank.load("missing")


def test_anatomy_bank_missing_parcellation_raises(tmp_path):
    anat_dir = tmp_path / "only_geom"
    (anat_dir / "parcellations").mkdir(parents=True)
    _write_geometry(anat_dir, N=10)
    bank = AnatomyBank(AnatomyBankConfig(bank_dir=tmp_path))
    with pytest.raises(FileNotFoundError, match="Parcellation not found"):
        bank.load("only_geom", scheme="never_saved")


def test_anatomy_bank_available_schemes(tmp_path):
    anat_dir = tmp_path / "multi"
    (anat_dir / "parcellations").mkdir(parents=True)
    _write_geometry(anat_dir, N=10)
    for scheme in ("desikan_killiany", "destrieux", "hcp_mmp1"):
        _write_parcellation(
            anat_dir / "parcellations", scheme,
            np.zeros(10, dtype=np.int32), ["x"],
        )
    bank = AnatomyBank(AnatomyBankConfig(bank_dir=tmp_path))
    assert bank.available_schemes("multi") == ["desikan_killiany", "destrieux", "hcp_mmp1"]
