"""Physics and coherence tests for pre-computed bank files.

All tests are skipped automatically when banks/ is not populated.
Run prepare_banks.py first to populate banks/, then re-run pytest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

BANKS_DIR = Path(__file__).parent.parent / "banks"


# ---------------------------------------------------------------------------
# Discovery helpers (pure filesystem scan, no MNE, no synthgen)
# ---------------------------------------------------------------------------

def _banks_present() -> bool:
    anat = BANKS_DIR / "anatomy"
    if not anat.exists():
        return False
    return any(d.is_dir() and (d / "source_space.npz").exists() for d in anat.iterdir())


def _get_anatomy_ids() -> list[str]:
    anat_dir = BANKS_DIR / "anatomy"
    if not anat_dir.exists():
        return []
    return [
        d.name for d in sorted(anat_dir.iterdir())
        if d.is_dir() and (d / "source_space.npz").exists()
    ]


def _is_bundle(npz: Path) -> bool:
    """A leadfield bundle has G + ch_names + electrode_coords. Anything else
    (e.g. legacy `electrode_coords.npz` or `standard.npz` from before the
    refactor) is filtered out so stale files don't break parametrization."""
    try:
        keys = set(np.load(npz, allow_pickle=False).files)
    except Exception:
        return False
    return {"G", "ch_names", "electrode_coords"}.issubset(keys)


def _get_leadfield_combos() -> list[tuple[str, str, str]]:
    lf_dir = BANKS_DIR / "leadfield"
    if not lf_dir.exists():
        return []
    combos = []
    for anat_dir in sorted(lf_dir.iterdir()):
        if not anat_dir.is_dir():
            continue
        for mont_dir in sorted(anat_dir.iterdir()):
            if not mont_dir.is_dir():
                continue
            for npz in sorted(mont_dir.glob("*.npz")):
                if not _is_bundle(npz):
                    continue
                combos.append((anat_dir.name, mont_dir.name, npz.stem))
    return combos


# ---------------------------------------------------------------------------
# Raw loaders (numpy only)
# ---------------------------------------------------------------------------

def _load_anatomy(anatomy_id: str):
    data = np.load(
        BANKS_DIR / "anatomy" / anatomy_id / "source_space.npz", allow_pickle=False
    )
    adj = sp.csr_matrix(
        (data["adjacency_data"], data["adjacency_indices"], data["adjacency_indptr"]),
        shape=tuple(data["adjacency_shape"]),
    )
    return data["vertex_coords"], adj, data["hemisphere"]


def _load_leadfield_bundle(anatomy_id: str, montage_id: str, conductivity_id: str):
    """Load (G, ch_names, electrode_coords) from a leadfield bundle NPZ."""
    data = np.load(
        BANKS_DIR / "leadfield" / anatomy_id / montage_id / f"{conductivity_id}.npz",
        allow_pickle=False,
    )
    return (
        data["G"],
        [str(s) for s in data["ch_names"]],
        data["electrode_coords"],
    )


# ---------------------------------------------------------------------------
# Parametrize lists (populated at import time; empty → skip sentinel)
# ---------------------------------------------------------------------------

_SKIP = pytest.mark.skip(reason="banks/ not populated — run prepare_banks.py first")
_ANATOMY_IDS = _get_anatomy_ids()
_LF_COMBOS = _get_leadfield_combos()

_ANATOMY_PARAMS = _ANATOMY_IDS or [pytest.param("_", marks=_SKIP)]
_LF_PARAMS = _LF_COMBOS or [pytest.param("_", "_", "_", marks=_SKIP)]


# ---------------------------------------------------------------------------
# Cross-coherence: G dims must match anatomy N and bundle C
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anatomy_id,montage_id,conductivity_id", _LF_PARAMS)
def test_leadfield_shape_matches_anatomy_and_montage(anatomy_id, montage_id, conductivity_id):
    G, ch_names, coords = _load_leadfield_bundle(anatomy_id, montage_id, conductivity_id)
    vertex_coords, _, _ = _load_anatomy(anatomy_id)

    assert G.shape[1] == vertex_coords.shape[0], (
        f"G N={G.shape[1]} != anatomy N={vertex_coords.shape[0]} "
        f"for {anatomy_id}/{montage_id}/{conductivity_id}"
    )
    assert G.shape[0] == len(ch_names), (
        f"G C={G.shape[0]} != ch_names C={len(ch_names)}"
    )
    assert G.shape[0] == coords.shape[0], (
        f"G C={G.shape[0]} != electrode_coords C={coords.shape[0]}"
    )


# ---------------------------------------------------------------------------
# Coordinate range: all coordinates must be within [-150, 200] mm (HEAD frame)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anatomy_id", _ANATOMY_PARAMS)
def test_vertex_coords_in_head_scale_mm(anatomy_id):
    vertex_coords, _, _ = _load_anatomy(anatomy_id)
    assert vertex_coords.min() >= -150.0, (
        f"{anatomy_id}: vertex_coords min {vertex_coords.min():.1f} mm < -150"
    )
    assert vertex_coords.max() <= 200.0, (
        f"{anatomy_id}: vertex_coords max {vertex_coords.max():.1f} mm > 200"
    )


@pytest.mark.parametrize("anatomy_id,montage_id,conductivity_id", _LF_PARAMS)
def test_electrode_coords_in_head_scale_mm(anatomy_id, montage_id, conductivity_id):
    _, _, coords = _load_leadfield_bundle(anatomy_id, montage_id, conductivity_id)
    assert coords.min() >= -150.0, (
        f"{anatomy_id}/{montage_id}: electrode coords min {coords.min():.1f} mm < -150"
    )
    assert coords.max() <= 200.0, (
        f"{anatomy_id}/{montage_id}: electrode coords max {coords.max():.1f} mm > 200"
    )


# ---------------------------------------------------------------------------
# Physics plausibility: column norms of G
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anatomy_id,montage_id,conductivity_id", _LF_PARAMS)
def test_leadfield_no_nan_inf(anatomy_id, montage_id, conductivity_id):
    G, _, _ = _load_leadfield_bundle(anatomy_id, montage_id, conductivity_id)
    assert not np.any(np.isnan(G)), f"NaN in G for {anatomy_id}/{montage_id}"
    assert not np.any(np.isinf(G)), f"Inf in G for {anatomy_id}/{montage_id}"


@pytest.mark.parametrize("anatomy_id,montage_id,conductivity_id", _LF_PARAMS)
def test_leadfield_no_zero_columns(anatomy_id, montage_id, conductivity_id):
    G, _, _ = _load_leadfield_bundle(anatomy_id, montage_id, conductivity_id)
    col_norms = np.linalg.norm(G, axis=0)
    n_zero = int((col_norms == 0).sum())
    assert n_zero == 0, (
        f"{anatomy_id}/{montage_id}: {n_zero} zero columns in G (silent sources)"
    )


@pytest.mark.parametrize("anatomy_id,montage_id,conductivity_id", _LF_PARAMS)
def test_leadfield_column_norm_ratio(anatomy_id, montage_id, conductivity_id):
    """max/mean column norm < 1e4: no single source dominates pathologically."""
    G, _, _ = _load_leadfield_bundle(anatomy_id, montage_id, conductivity_id)
    col_norms = np.linalg.norm(G, axis=0)
    ratio = col_norms.max() / (col_norms.mean() + 1e-30)
    assert ratio < 1e4, (
        f"{anatomy_id}/{montage_id}: col_norm max/mean ratio {ratio:.2e} >= 1e4"
    )


@pytest.mark.parametrize("anatomy_id,montage_id,conductivity_id", _LF_PARAMS)
def test_leadfield_spatial_variation(anatomy_id, montage_id, conductivity_id):
    """Column norm CV > 0.05: leadfield must not be spatially uniform.

    Both vertex_coords and electrode_coords are now in the same HEAD frame,
    but the column-norm CV check is a frame-agnostic plausibility test.
    """
    G, _, _ = _load_leadfield_bundle(anatomy_id, montage_id, conductivity_id)
    col_norms = np.linalg.norm(G, axis=0)
    cv = col_norms.std() / (col_norms.mean() + 1e-30)
    assert cv > 0.05, (
        f"{anatomy_id}/{montage_id}: col_norm CV={cv:.4f} < 0.05 — leadfield looks spatially uniform"
    )


# ---------------------------------------------------------------------------
# Anatomy consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anatomy_id", _ANATOMY_PARAMS)
def test_anatomy_both_hemispheres_present(anatomy_id):
    _, _, hemisphere = _load_anatomy(anatomy_id)
    n_lh = int((hemisphere == 0).sum())
    n_rh = int((hemisphere == 1).sum())
    assert n_lh > 0, f"{anatomy_id}: no left-hemisphere vertices"
    assert n_rh > 0, f"{anatomy_id}: no right-hemisphere vertices"


@pytest.mark.parametrize("anatomy_id", _ANATOMY_PARAMS)
def test_anatomy_hemispheres_roughly_balanced(anatomy_id):
    """Neither hemisphere should have more than 3x the vertices of the other."""
    _, _, hemisphere = _load_anatomy(anatomy_id)
    n_lh = int((hemisphere == 0).sum())
    n_rh = int((hemisphere == 1).sum())
    ratio = max(n_lh, n_rh) / (min(n_lh, n_rh) + 1)
    assert ratio < 3.0, (
        f"{anatomy_id}: hemisphere imbalance L={n_lh} R={n_rh} (ratio {ratio:.1f})"
    )


@pytest.mark.parametrize("anatomy_id", _ANATOMY_PARAMS)
def test_anatomy_adjacency_symmetric(anatomy_id):
    _, adjacency, _ = _load_anatomy(anatomy_id)
    diff = adjacency - adjacency.T
    diff.eliminate_zeros()
    assert diff.nnz == 0, f"{anatomy_id}: adjacency is not symmetric"


# ---------------------------------------------------------------------------
# Coordinate frame alignment: vertex_coords and electrode_coords same frame
# ---------------------------------------------------------------------------

def _load_electrode_coords(anatomy_id: str, montage_id: str) -> np.ndarray:
    """Load anatomy-specific electrode coords (same head frame as vertex_coords).

    Prefers leadfield/{anatomy_id}/{montage_id}/electrode_coords.npz (saved from
    fwd["info"]["chs"], same head frame as fwd["src"]["rr"]). Falls back to the
    standard montage bank if the anatomy-specific file is absent.
    """
    elec_path = BANKS_DIR / "leadfield" / anatomy_id / montage_id / "electrode_coords.npz"
    if elec_path.exists():
        return np.load(elec_path, allow_pickle=False)["coords"]
    data = np.load(BANKS_DIR / "montage" / f"{montage_id}.npz", allow_pickle=False)
    return data["coords"]


@pytest.mark.parametrize("anatomy_id,montage_id,conductivity_id", _LF_PARAMS)
def test_vertex_and_electrode_centroids_aligned(anatomy_id, montage_id, conductivity_id):
    """Centroid of vertex_coords and electrode_coords must be within 110 mm.

    Both are in the anatomy's head frame (from the forward solution). Sparse
    montages (21-32 ch) bias the electrode centroid frontally — observed max
    ~90 mm in correct frame. A true frame mismatch gives 150+ mm.
    """
    vertex_coords, _, _ = _load_anatomy(anatomy_id)
    coords = _load_electrode_coords(anatomy_id, montage_id)
    vc_centroid = vertex_coords.mean(axis=0)
    el_centroid = coords.mean(axis=0)
    dist = float(np.linalg.norm(vc_centroid - el_centroid))
    assert dist < 110.0, (
        f"{anatomy_id}/{montage_id}: centroid distance {dist:.1f} mm >= 110 mm "
        f"— vertex_coords and electrode_coords may be in different coordinate frames"
    )
