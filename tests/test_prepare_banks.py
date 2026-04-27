import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp


def _load_prepare_banks():
    spec = importlib.util.spec_from_file_location(
        "prepare_banks",
        Path(__file__).parent.parent / "scripts" / "prepare_banks.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prepare_banks_is_importable():
    mod = _load_prepare_banks()
    assert hasattr(mod, "main")
    assert hasattr(mod, "MONTAGE_MAP")


def test_montage_map_covers_all_config_montages():
    mod = _load_prepare_banks()
    expected = {
        "standard_1005_21", "standard_1005_32", "standard_1005_64",
        "standard_1005_76", "standard_1005_90", "standard_1005_128", "standard_1005_256",
    }
    assert expected.issubset(set(mod.MONTAGE_MAP.keys()))


def test_save_and_load_source_space_roundtrip(tmp_path):
    mod = _load_prepare_banks()
    N = 30
    vertex_coords = np.random.randn(N, 3).astype(np.float32)
    adjacency = sp.eye(N, format="csr", dtype=np.float32)
    parcellation = np.zeros(N, dtype=np.int32)
    hemisphere = np.zeros(N, dtype=np.int32)

    anatomy_dir = tmp_path / "anatomy" / "test_subject"
    (anatomy_dir / "parcellations").mkdir(parents=True)
    mod._save_source_space(
        anatomy_dir / "source_space.npz", vertex_coords, adjacency, hemisphere,
    )
    mod._save_parcellation(
        anatomy_dir / "parcellations" / "desikan_killiany.npz",
        parcellation, ["A", "B"], scheme="desikan_killiany",
    )

    from synthgen.banks.anatomy import AnatomyBank
    from synthgen.config import AnatomyBankConfig
    bank = AnatomyBank(AnatomyBankConfig(bank_dir=tmp_path / "anatomy"))
    ss = bank.load("test_subject")
    assert ss.vertex_coords.shape == (N, 3)
    assert ss.adjacency.shape == (N, N)
    assert ss.parcellation.shape == (N,)
    assert ss.hemisphere.shape == (N,)
    assert ss.parcellation_scheme == "desikan_killiany"
    assert ss.region_labels == ["A", "B"]


def test_anatomy_bank_multi_scheme(tmp_path):
    mod = _load_prepare_banks()
    N = 30
    anatomy_dir = tmp_path / "anatomy" / "test_subject"
    (anatomy_dir / "parcellations").mkdir(parents=True)
    mod._save_source_space(
        anatomy_dir / "source_space.npz",
        np.random.randn(N, 3).astype(np.float32),
        sp.eye(N, format="csr", dtype=np.float32),
        np.zeros(N, dtype=np.int32),
    )
    mod._save_parcellation(
        anatomy_dir / "parcellations" / "desikan_killiany.npz",
        np.zeros(N, dtype=np.int32), ["dk0", "dk1"], scheme="desikan_killiany",
    )
    mod._save_parcellation(
        anatomy_dir / "parcellations" / "destrieux.npz",
        np.arange(N, dtype=np.int32) % 5,
        [f"d{i}" for i in range(5)], scheme="destrieux",
    )

    from synthgen.banks.anatomy import AnatomyBank
    from synthgen.config import AnatomyBankConfig
    bank = AnatomyBank(AnatomyBankConfig(bank_dir=tmp_path / "anatomy"))
    dk = bank.load("test_subject", scheme="desikan_killiany")
    de = bank.load("test_subject", scheme="destrieux")
    assert dk.parcellation_scheme == "desikan_killiany"
    assert de.parcellation_scheme == "destrieux"
    assert len(dk.region_labels) == 2
    assert len(de.region_labels) == 5
    # Geometry is shared
    assert dk.vertex_coords.shape == de.vertex_coords.shape


def test_anatomy_bank_missing_parcellation_raises(tmp_path):
    mod = _load_prepare_banks()
    N = 10
    anatomy_dir = tmp_path / "anatomy" / "test_subject"
    (anatomy_dir / "parcellations").mkdir(parents=True)
    mod._save_source_space(
        anatomy_dir / "source_space.npz",
        np.zeros((N, 3), dtype=np.float32),
        sp.eye(N, format="csr", dtype=np.float32),
        np.zeros(N, dtype=np.int32),
    )

    from synthgen.banks.anatomy import AnatomyBank
    from synthgen.config import AnatomyBankConfig
    bank = AnatomyBank(AnatomyBankConfig(bank_dir=tmp_path / "anatomy"))
    with pytest.raises(FileNotFoundError, match="Parcellation not found"):
        bank.load("test_subject", scheme="unknown_scheme")


def test_save_and_load_leadfield_roundtrip(tmp_path):
    mod = _load_prepare_banks()
    G = np.random.randn(64, 100).astype(np.float32)
    path = tmp_path / "test_lf.npz"
    mod._save_leadfield(path, G)
    loaded = np.load(path, allow_pickle=False)["G"]
    assert loaded.shape == (64, 100)
    assert np.allclose(loaded, G)


def test_save_and_load_montage_roundtrip(tmp_path):
    mod = _load_prepare_banks()
    coords = np.random.randn(64, 3).astype(np.float32)
    ch_names = [f"Ch{i:03d}" for i in range(64)]
    path = tmp_path / "test_montage.npz"
    mod._save_montage(path, coords, ch_names)

    from synthgen.banks.montage import MontageBank
    from synthgen.config import MontageBankConfig
    bank = MontageBank(MontageBankConfig(bank_dir=tmp_path, montages=[]))
    m = bank.load("test_montage")
    assert m.coords.shape == (64, 3)
    assert len(m.ch_names) == 64
    assert m.ch_names[0] == "Ch000"


# ---------------------------------------------------------------------------
# Structural invariants (synthetic data, no MNE, no disk banks)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,C", [(10, 5), (100, 64), (20484, 128)])
def test_source_space_dtype_roundtrip(tmp_path, N, C):
    mod = _load_prepare_banks()
    rng = np.random.default_rng(0)
    vertex_coords = rng.standard_normal((N, 3)).astype(np.float64)
    adjacency = sp.eye(N, format="csr", dtype=np.float64)
    hemisphere = np.zeros(N, dtype=np.int64)

    out = tmp_path / "source_space.npz"
    mod._save_source_space(out, vertex_coords, adjacency, hemisphere)
    data = np.load(out, allow_pickle=False)

    assert data["vertex_coords"].dtype == np.float32
    assert data["vertex_coords"].shape == (N, 3)
    assert data["adjacency_data"].dtype == np.float32
    assert data["adjacency_indices"].dtype == np.int32
    assert data["adjacency_indptr"].dtype == np.int32
    assert data["hemisphere"].dtype == np.int32
    assert data["hemisphere"].shape == (N,)


def test_leadfield_dtype_and_nan_invariants(tmp_path):
    mod = _load_prepare_banks()
    rng = np.random.default_rng(1)
    G = rng.standard_normal((64, 1000)).astype(np.float32)
    path = tmp_path / "G.npz"
    mod._save_leadfield(path, G)
    loaded = np.load(path, allow_pickle=False)["G"]
    assert loaded.dtype == np.float32
    assert loaded.shape == (64, 1000)
    assert not np.any(np.isnan(loaded))
    assert not np.any(np.isinf(loaded))


def test_montage_dtype_and_nan_invariants(tmp_path):
    mod = _load_prepare_banks()
    rng = np.random.default_rng(2)
    coords = rng.standard_normal((64, 3)).astype(np.float32)
    ch_names = [f"Ch{i:03d}" for i in range(64)]
    path = tmp_path / "montage.npz"
    mod._save_montage(path, coords, ch_names)
    data = np.load(path, allow_pickle=False)
    assert data["coords"].dtype == np.float32
    assert data["coords"].shape == (64, 3)
    assert not np.any(np.isnan(data["coords"]))
    assert not np.any(np.isinf(data["coords"]))
    assert len(data["ch_names"]) == 64


def test_adjacency_symmetry_preserved(tmp_path):
    mod = _load_prepare_banks()
    N = 20
    rng = np.random.default_rng(3)
    A = sp.random(N, N, density=0.3, format="csr", random_state=rng, dtype=np.float32)
    A = (A + A.T)
    A.setdiag(1.0)

    out = tmp_path / "ss.npz"
    mod._save_source_space(
        out, np.zeros((N, 3), dtype=np.float32), A, np.zeros(N, dtype=np.int32),
    )
    data = np.load(out, allow_pickle=False)
    loaded = sp.csr_matrix(
        (data["adjacency_data"], data["adjacency_indices"], data["adjacency_indptr"]),
        shape=tuple(data["adjacency_shape"]),
    )
    diff = loaded - loaded.T
    diff.eliminate_zeros()
    assert diff.nnz == 0, "Adjacency must be symmetric after save/load"


def test_no_isolated_vertices_preserved(tmp_path):
    mod = _load_prepare_banks()
    N = 10
    rows = list(range(N - 1)) + list(range(1, N))
    cols = list(range(1, N)) + list(range(N - 1))
    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, N)
    )
    out = tmp_path / "ss.npz"
    mod._save_source_space(
        out, np.zeros((N, 3), dtype=np.float32), A, np.zeros(N, dtype=np.int32),
    )
    data = np.load(out, allow_pickle=False)
    loaded = sp.csr_matrix(
        (data["adjacency_data"], data["adjacency_indices"], data["adjacency_indptr"]),
        shape=tuple(data["adjacency_shape"]),
    )
    row_nnz = np.diff(loaded.indptr)
    assert np.all(row_nnz >= 1), "No vertex should be isolated (0 neighbours)"


def test_hemisphere_values_binary(tmp_path):
    mod = _load_prepare_banks()
    N = 20
    hemisphere = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    out = tmp_path / "ss.npz"
    mod._save_source_space(
        out, np.zeros((N, 3), dtype=np.float32),
        sp.eye(N, format="csr", dtype=np.float32), hemisphere,
    )
    data = np.load(out, allow_pickle=False)
    unique_vals = set(np.unique(data["hemisphere"]).tolist())
    assert unique_vals.issubset({0, 1}), f"hemisphere values not in {{0,1}}: {unique_vals}"


def test_parcellation_save_roundtrip(tmp_path):
    mod = _load_prepare_banks()
    N = 20
    parcellation = np.arange(N, dtype=np.int32)
    labels = [f"r{i}" for i in range(N)]
    out = tmp_path / "dk.npz"
    mod._save_parcellation(out, parcellation, labels, scheme="desikan_killiany")
    data = np.load(out, allow_pickle=False)
    assert data["parcellation"].dtype == np.int32
    assert np.all(data["parcellation"] >= 0)
    assert str(data["scheme"]) == "desikan_killiany"
    assert len(data["region_labels"]) == N


def test_read_deepsif_picks_rm_over_object_neighbours(tmp_path):
    """The DeepSIF .mat ships two arrays > 100 elements: 'rm' (uint16 region
    mapping, length 20484) and 'nbs' (object, length 994 lists of variable
    length). Earlier code picked candidates[0] which on some Python builds
    landed on 'nbs' and failed with 'setting an array element with a
    sequence'. This test writes a fake .mat with the same ambiguous shape
    and verifies 'rm' wins."""
    from scipy.io import savemat

    mod = _load_prepare_banks()
    n_vertices = 20484
    n_regions = 994
    rm = (np.arange(n_vertices) % n_regions).astype(np.uint16) + 1  # MATLAB 1-indexed
    nbs = np.empty((1, n_regions), dtype=object)
    for i in range(n_regions):
        nbs[0, i] = np.array(
            [(i + 1) % n_regions, (i + 2) % n_regions], dtype=np.uint16
        )
    deepsif_dir = tmp_path / "deepsif"
    deepsif_dir.mkdir()
    savemat(deepsif_dir / "fs_cortex_20k_region_mapping.mat", {"nbs": nbs, "rm": rm})

    # Fake MNE source space stub: only needs sum(len(s["vertno"])) == 20484
    src = [
        {"vertno": np.arange(10242, dtype=np.int64), "inuse": np.ones(10242, dtype=int)},
        {"vertno": np.arange(10242, dtype=np.int64), "inuse": np.ones(10242, dtype=int)},
    ]

    parc, labels = mod._read_deepsif_parcellation(deepsif_dir, src)
    assert parc.shape == (n_vertices,)
    assert parc.dtype == np.int32
    assert int(parc.min()) == 0  # 1-indexed input -> 0-indexed output
    assert int(parc.max()) == n_regions - 1
    assert len(labels) == n_regions
