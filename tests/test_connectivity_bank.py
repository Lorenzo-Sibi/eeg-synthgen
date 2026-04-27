import numpy as np
import pytest
from synthgen.banks.connectivity import ConnectivityBank
from synthgen.config import ConnectivityBankConfig


def _write_fake_conn(path, R=4):
    np.savez_compressed(
        path,
        weights=np.eye(R, dtype=np.float32),
        tract_lengths=np.ones((R, R), dtype=np.float32) * 10.0,
        region_centers=np.zeros((R, 3), dtype=np.float32),
        region_labels=np.array([f"r{i}" for i in range(R)], dtype=str),
        scheme="test",
    )


def test_connectivity_load(tmp_path):
    _write_fake_conn(tmp_path / "test.npz")
    bank = ConnectivityBank(ConnectivityBankConfig(bank_dir=tmp_path))
    c = bank.load("test")
    assert c.weights.shape == (4, 4)
    assert c.tract_lengths.shape == (4, 4)
    assert c.region_centers.shape == (4, 3)
    assert c.region_labels == ["r0", "r1", "r2", "r3"]
    assert c.scheme == "test"


def test_connectivity_missing_raises(tmp_path):
    bank = ConnectivityBank(ConnectivityBankConfig(bank_dir=tmp_path))
    with pytest.raises(FileNotFoundError):
        bank.load("nope")


def test_connectivity_cache(tmp_path):
    _write_fake_conn(tmp_path / "test.npz")
    bank = ConnectivityBank(ConnectivityBankConfig(bank_dir=tmp_path))
    c1 = bank.load("test")
    c2 = bank.load("test")
    assert c1 is c2