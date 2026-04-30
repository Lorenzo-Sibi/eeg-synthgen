"""Offline bank preparation: compute and save anatomy, leadfield, and montage banks.

Usage:
    # Prepare all anatomies listed in config (skips nyhead unless --nyhead-mat is given)
    python scripts/prepare_banks.py --config config/default.yaml

    # Prepare a specific anatomy only
    python scripts/prepare_banks.py --config config/default.yaml --anatomy fsaverage

    # Control source-space resolution (oct4=~2k, oct5=~8k, oct6=~32k vertices)
    python scripts/prepare_banks.py --config config/default.yaml --spacing oct5

    # NY Head requires the external .mat file (download from parralab.org/nyhead)
    python scripts/prepare_banks.py --config config/default.yaml \\
        --anatomy nyhead --nyhead-mat /path/to/sa_nyhead.mat

Bank file final layout
----------------
banks/
  anatomy/{anatomy_id}/source_space.npz
  leadfield/{anatomy_id}/{montage_id}/{conductivity_id}.npz    (G matrix CxN)
  montage/{montage_id}.npz                                     (coords + ch_names)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import mne
from mne.transforms import apply_trans
import scipy.sparse as sp

from synthgen.config import GenerationConfig

# Maps internal montage_id -> (MNE montage name, channel count or None=keep all)
MONTAGE_MAP: dict[str, tuple[str, int | None]] = {
    "standard_1005_21":  ("standard_1005", 21),
    "standard_1005_32":  ("standard_1005", 32),
    "standard_1005_64":  ("standard_1005", 64),
    "standard_1005_76":  ("standard_1005", 76),
    "standard_1005_90":  ("standard_1005", 90),
    "standard_1005_128": ("standard_1005", 128),
    "standard_1005_256": ("standard_1005", 256),
}

def _save_source_space(
    path: Path,
    vertex_coords: np.ndarray,
    adjacency: sp.spmatrix,
    hemisphere: np.ndarray,
) -> None:
    """Save source-space geometry (vertex_coords, adjacency, hemisphere).

    Parcellation is stored in a separate per-scheme file under ``parcellations/``
    so the same geometry can back multiple parcellations without duplication.
    """
    adj = adjacency.tocsr()
    np.savez_compressed(
        path,
        vertex_coords=vertex_coords.astype(np.float32),
        adjacency_data=adj.data.astype(np.float32),
        adjacency_indices=adj.indices.astype(np.int32),
        adjacency_indptr=adj.indptr.astype(np.int32),
        adjacency_shape=np.array(adj.shape, dtype=np.int32),
        hemisphere=hemisphere.astype(np.int32),
    )


def _save_parcellation(
    path: Path,
    parcellation: np.ndarray,
    region_labels: list[str],
    scheme: str,
) -> None:
    """Save a parcellation array + labels for a given scheme."""
    np.savez_compressed(
        path,
        parcellation=parcellation.astype(np.int32),
        region_labels=np.array(region_labels, dtype=str),
        scheme=scheme,
    )


def _save_leadfield(
    path: Path,
    G: np.ndarray,
    ch_names: list[str],
    electrode_coords: np.ndarray,
) -> None:
    """Save a leadfield bundle: G + ch_names + electrode_coords (HEAD frame)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        G=G.astype(np.float32),
        ch_names=np.array(ch_names, dtype=str),
        electrode_coords=electrode_coords.astype(np.float32),
    )

# TODO: change this and implement Andrea's idea of kmean-plusplus
def _select_distributed_channels(
    ch_pos: dict[str, np.ndarray],
    n: int,
) -> list[str]:
    """Select n channels from ch_pos with maximum scalp coverage.

    Strategy: uses farthest-point (maximin) sampling: seed at the electrode closest to
    the crown of the head (highest z in MNE head frame), then iteratively add the 
    electrode that maximises the minimum distance to all already-selected electrodes.

    Time complexity: O(C * n) - it's ok for small montages
    """
    names  = list(ch_pos.keys())
    coords = np.array([ch_pos[name] for name in names], dtype=np.float64)  # (C, 3) metres

    if n >= len(names):
        return names

    # Seed: electrode closest to the crown (0, 0, z_max) — typically Cz
    crown    = np.array([0.0, 0.0, coords[:, 2].max()])
    seed_idx = int(np.argmin(np.linalg.norm(coords - crown, axis=1)))

    selected  = [seed_idx]
    min_dists = np.linalg.norm(coords - coords[seed_idx], axis=1)  # (C,)

    while len(selected) < n:
        next_idx  = int(np.argmax(min_dists))
        selected.append(next_idx)
        d         = np.linalg.norm(coords - coords[next_idx], axis=1)
        min_dists = np.minimum(min_dists, d)

    return [names[i] for i in selected]


# Maps parcellation scheme -> MNE annot name. Populated for schemes that are
# a thin wrapper around ``mne.read_labels_from_annot``.
_MNE_ANNOT_NAMES: dict[str, str] = {
    "desikan_killiany": "aparc",
    "destrieux": "aparc.a2009s",
    "hcp_mmp1": "HCPMMP1",
    "schaefer_400": "Schaefer2018_400Parcels_17Networks_order",
    "schaefer_1000": "Schaefer2018_1000Parcels_17Networks_order",
}


def _read_mne_annot_parcellation(
    annot: str,
    subject: str,
    subjects_dir: Path,
    src: mne.SourceSpaces,
    inuse_src: mne.SourceSpaces | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return dense 0..R-1 parcellation (N,) and R labels from an MNE .annot file.

    If ``inuse_src`` is provided, only vertices active in its ``inuse`` mask are
    kept — so the result aligns with the geometry saved by
    ``_extract_source_space(src, inuse_src=fwd_src)``. Unlabeled vertices are
    assigned to a dedicated 'unknown' bucket, and empty parcels are compacted.
    """
    labels = mne.read_labels_from_annot(
        subject, parc=annot, subjects_dir=str(subjects_dir), verbose=False
    )
    label_names = [lab.name for lab in labels]

    src_vertnos = [s["vertno"] for s in src]
    n_total = sum(len(v) for v in src_vertnos)
    parc = -np.ones(n_total, dtype=np.int64)

    offset = 0
    for hemi_str, vertno in zip(["lh", "rh"], src_vertnos):
        for li, lab in enumerate(labels):
            if lab.hemi != hemi_str:
                continue
            mask = np.isin(vertno, lab.vertices)
            parc[offset + np.where(mask)[0]] = li
        offset += len(vertno)

    if inuse_src is not None:
        keep = np.concatenate(
            [inuse_src[i]["inuse"][src[i]["vertno"]].astype(bool) for i in (0, 1)]
        )
        parc = parc[keep]

    if np.any(parc < 0):
        unknown_id = len(labels)
        parc[parc < 0] = unknown_id
        label_names = label_names + ["unknown"]

    used_ids = np.unique(parc)
    remap = -np.ones(len(label_names), dtype=np.int64)
    remap[used_ids] = np.arange(len(used_ids))
    parc = remap[parc].astype(np.int32)
    label_names = [label_names[i] for i in used_ids.tolist()]
    return parc, label_names


def _read_deepsif_parcellation(
    deepsif_anatomy_dir: Path,
    src: mne.SourceSpaces,
    inuse_src: mne.SourceSpaces | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load DeepSIF-994 parcellation from ``fs_cortex_20k_region_mapping.mat``.

    DeepSIF uses fsaverage5 (20484 vertices, same as MNE's ``ico5``). The mapping
    is a (20484,) int vector of region ids. Requires the source space to be
    fsaverage5 (ico5).
    """
    from scipy.io import loadmat

    mapping_path = deepsif_anatomy_dir / "fs_cortex_20k_region_mapping.mat"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"DeepSIF region mapping not found at {mapping_path}. "
            "Run scripts/fetch_atlases.py --deepsif first."
        )
    m = loadmat(str(mapping_path))
    # The DeepSIF .mat ships two arrays: 'rm' (region mapping, uint16, length 20484) 
    # and 'nbs' (neighbourhood lists, dtype=object, length 994).
    candidate = m["rm"] if "rm" in m and isinstance(m["rm"], np.ndarray) else None
    if candidate is None:
        raise ValueError(f"No region-mapping array found in {mapping_path}")
    region_map = np.asarray(candidate).squeeze().astype(np.int64)

    n_src = sum(len(s["vertno"]) for s in src)
    if region_map.size != n_src:
        raise ValueError(
            f"DeepSIF mapping has {region_map.size} entries but source space has "
            f"{n_src} vertices. DeepSIF requires ico5 (20484 vertices)."
        )

    if int(region_map.min()) == 1:
        print("  Adjusting DeepSIF region ids from 1-indexed to 0-indexed...")
        region_map = region_map - 1  # MATLAB 1-indexed -> 0-indexed

    if inuse_src is not None:
        keep = np.concatenate(
            [inuse_src[i]["inuse"][src[i]["vertno"]].astype(bool) for i in (0, 1)]
        )
        region_map = region_map[keep]

    used_ids = np.unique(region_map)
    remap = -np.ones(int(region_map.max()) + 1, dtype=np.int64)
    remap[used_ids] = np.arange(len(used_ids))
    parc = remap[region_map].astype(np.int32)
    label_names = [f"region_{i:04d}" for i in used_ids.tolist()]
    return parc, label_names


def _read_parcellation(
    scheme: str,
    subject: str,
    subjects_dir: Path,
    src: mne.SourceSpaces,
    deepsif_anatomy_dir: Path | None = None,
    inuse_src: mne.SourceSpaces | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Dispatch to the right parcellation reader for ``scheme``."""
    if scheme in _MNE_ANNOT_NAMES:
        return _read_mne_annot_parcellation(
            _MNE_ANNOT_NAMES[scheme], subject, subjects_dir, src, inuse_src=inuse_src,
        )
    if scheme == "deepsif_994":
        if deepsif_anatomy_dir is None:
            raise ValueError(
                "deepsif_994 requires --deepsif-dir pointing to the DeepSIF anatomy folder"
            )
        return _read_deepsif_parcellation(deepsif_anatomy_dir, src, inuse_src=inuse_src)
    raise ValueError(f"Unknown parcellation scheme: {scheme!r}")


def _extract_source_space(
    fwd_src: list,
) -> tuple[np.ndarray, sp.spmatrix, np.ndarray]:
    """Extract geometry (vertex_coords mm HEAD frame, adjacency, hemisphere)
    from a forward source space.

    ``fwd_src`` is the post-forward source space (``fwd["src"]``) — its
    ``inuse`` mask reflects BEM-pruning, its ``rr`` is already in HEAD frame
    (``coord_frame == 4``), so no transform is applied here. Always pass
    ``fwd["src"]`` rather than the original ``setup_source_space`` output;
    otherwise the inuse mask and the coordinate frame disagree.

    Returns
    -------
    vertex_coords : (N, 3) float32, mm, HEAD frame.
    adjacency :    (N, N) sparse — cortical neighbourhood graph from MNE.
    hemisphere :   (N,)   int32  — 0 = LH, 1 = RH.
    """
    adjacency = mne.spatial_src_adjacency(fwd_src, verbose=False)
    lh_used = fwd_src[0]["inuse"].astype(bool)
    rh_used = fwd_src[1]["inuse"].astype(bool)
    lh_coords = fwd_src[0]["rr"][lh_used] * 1000.0
    rh_coords = fwd_src[1]["rr"][rh_used] * 1000.0
    vertex_coords = np.concatenate([lh_coords, rh_coords], axis=0).astype(np.float32)

    n_lh = int(lh_used.sum())
    n_rh = int(rh_used.sum())
    hemisphere = np.concatenate(
        [np.zeros(n_lh, dtype=np.int32), np.ones(n_rh, dtype=np.int32)]
    )
    return vertex_coords, adjacency, hemisphere


def _build_dig_montage(
    montage_id: str,
) -> tuple[mne.channels.DigMontage, list[str], np.ndarray]:
    """Build an MNE DigMontage for a given montage_id.

    Pure builder: no file IO. Returns the DigMontage, the selected channel
    names (subset preserving the standard 10-05 layout), and the raw montage
    coords in mm. Persistence is the orchestrator's responsibility.
    """
    mne_name, n_channels = MONTAGE_MAP[montage_id]
    raw_montage = mne.channels.make_standard_montage(mne_name)
    positions = raw_montage.get_positions()
    coord_frame = positions["coord_frame"]
    ch_pos = positions["ch_pos"]
    nasion = positions.get("nasion")
    lpa = positions.get("lpa")
    rpa = positions.get("rpa")

    ch_names = (
        _select_distributed_channels(ch_pos, n_channels)
        if n_channels is not None and len(ch_pos) > n_channels
        else list(ch_pos.keys())
    )
    ch_pos_subset = {ch: ch_pos[ch] for ch in ch_names}
    coords_mm = np.array(
        [ch_pos[ch] for ch in ch_names], dtype=np.float32,
    ) * 1000.0

    dig_montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos_subset,
        coord_frame=coord_frame,
        nasion=nasion,
        lpa=lpa,
        rpa=rpa,
    )
    return dig_montage, ch_names, coords_mm


def _compute_forward(
    info: mne.Info,
    src: list,
    bem_sol: Any,
    trans: str | Path,
) -> mne.Forward:
    """Compute the fixed-orientation EEG forward solution for one montage.

    Returns the fwd object as-is. Callers index into ``fwd["sol"]["data"]``
    for G, ``fwd["info"]["chs"]`` for HEAD-frame electrodes,
    ``fwd["src"]`` for the post-pruning source space, and
    ``fwd["mri_head_t"]`` for the head-frame transform.
    """
    fwd = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem_sol,
        meg=False, eeg=True, n_jobs=-1, verbose=False,
    )
    return mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, verbose=False,
    )


def _assert_same_pruning(fwd_src: list, canonical: list) -> None:
    """Raise if the BEM-pruning differs between two forward source spaces.

    BEM-pruning is determined by ``(src, bem, trans, mindist)`` — sensors do
    not influence which sources survive. Catches MNE behavior changes early.
    """
    for hemi, (a, b) in enumerate(zip(fwd_src, canonical)):
        if not np.array_equal(a["inuse"], b["inuse"]):
            raise RuntimeError(
                f"Forward solution prunes hemisphere {hemi} differently "
                "across montages — BEM-pruning should be montage-independent. "
                "Check MNE version / mindist."
            )


def prepare_mne_anatomy(
    anatomy_id: str,
    subject: str,
    subjects_dir: Path,
    bem_sol: Any,
    trans: str | Path,
    spacing: str,
    montage_ids: list[str],
    parcellation_schemes: list[str],
    conductivity_id: str,
    anatomy_bank_dir: Path,
    leadfield_bank_dir: Path,
    deepsif_anatomy_dir: Path | None = None,
) -> None:
    """Shared orchestrator for MNE-based anatomies (fsaverage, mne_sample).

    The workflow:
      1. Set up the source space
      2. For each montage: build dig montage, compute forward, write the
         leadfield bundle. The first forward's ``fwd["src"]`` becomes the
         canonical source space for parcellation.
      3. Save anatomy geometry using the canonical source space (``rr`` is already in HEAD)
      4. Save each parcellation restricted to the canonical source space.
    """
    print(f"  Setting up source space (spacing={spacing})...")
    src = mne.setup_source_space(
        subject, spacing=spacing, subjects_dir=subjects_dir,
        add_dist="patch", verbose=False,
    )

    anatomy_dir = anatomy_bank_dir / anatomy_id
    (anatomy_dir / "parcellations").mkdir(parents=True, exist_ok=True)

    fwd_src_canonical: list | None = None

    for montage_id in montage_ids:
        if montage_id not in MONTAGE_MAP:
            print(f"  Skipping unknown montage: {montage_id}")
            continue

        dig_montage, ch_names, _ = _build_dig_montage(montage_id)
        info = mne.create_info(dig_montage.ch_names, sfreq=500, ch_types="eeg")
        info.set_montage(dig_montage, on_missing="raise", verbose=False)

        print(f"    Computing forward solution ({montage_id}, {len(ch_names)} ch)...")
        fwd = _compute_forward(info, src, bem_sol, trans)

        if fwd_src_canonical is None:
            fwd_src_canonical = fwd["src"]
        else:
            _assert_same_pruning(fwd["src"], fwd_src_canonical)

        G = fwd["sol"]["data"]
        elec_coords_mm = np.array(
            [ch["loc"][:3] for ch in fwd["info"]["chs"]], dtype=np.float32,
        ) * 1000.0

        lf_path = (
            leadfield_bank_dir / anatomy_id / montage_id / f"{conductivity_id}.npz"
        )
        _save_leadfield(lf_path, G, ch_names, elec_coords_mm)
        print(f"    Saved leadfield {anatomy_id}/{montage_id}/{conductivity_id}: {G.shape}")

    if fwd_src_canonical is None:
        raise RuntimeError(
            f"No valid montages were processed for anatomy {anatomy_id!r}; "
            "cannot save source space."
        )

    vertex_coords, adjacency, hemisphere = _extract_source_space(fwd_src_canonical)
    _save_source_space(anatomy_dir / "source_space.npz", vertex_coords, adjacency, hemisphere)
    
    print(f"  Saved source space: {len(vertex_coords)} vertices (HEAD frame)")

    for scheme in parcellation_schemes:
        try:
            parc, label_names = _read_parcellation(
                scheme, subject, subjects_dir, src,
                deepsif_anatomy_dir=deepsif_anatomy_dir,
                inuse_src=fwd_src_canonical,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"    Skipping parcellation {scheme!r}: {e}")
            continue
        _save_parcellation(
            anatomy_dir / "parcellations" / f"{scheme}.npz",
            parc, label_names, scheme,
        )
        print(f"    Saved parcellation {scheme}: {len(label_names)} regions")


def prepare_fsaverage(
    anatomy_bank_dir: Path,
    leadfield_bank_dir: Path,
    montage_ids: list[str],
    spacing: str,
    parcellation_schemes: list[str],
    conductivity_id: str = "standard",
    deepsif_anatomy_dir: Path | None = None,
) -> None:
    """Prepare anatomy + leadfield banks for the MNE fsaverage subject."""
    subjects_dir = Path(mne.datasets.fetch_fsaverage(verbose=False)).parent

    print("  Building BEM model (3-shell, ico=4)...")
    bem_model = mne.make_bem_model(
        subject="fsaverage", subjects_dir=subjects_dir,
        ico=4, conductivity=[0.3, 0.006, 0.3], verbose=False,
    )
    bem_sol = mne.make_bem_solution(bem_model, verbose=False)

    prepare_mne_anatomy(
        anatomy_id="fsaverage",
        subject="fsaverage",
        subjects_dir=subjects_dir,
        bem_sol=bem_sol,
        trans="fsaverage",
        spacing=spacing,
        montage_ids=montage_ids,
        parcellation_schemes=parcellation_schemes,
        conductivity_id=conductivity_id,
        anatomy_bank_dir=anatomy_bank_dir,
        leadfield_bank_dir=leadfield_bank_dir,
        deepsif_anatomy_dir=deepsif_anatomy_dir,
    )


def prepare_mne_sample(
    anatomy_bank_dir: Path,
    leadfield_bank_dir: Path,
    montage_ids: list[str],
    spacing: str,
    parcellation_schemes: list[str],
    conductivity_id: str = "standard",
    deepsif_anatomy_dir: Path | None = None,
) -> None:
    """Prepare anatomy + leadfield banks for the MNE sample subject.

    The MNE sample dataset includes a real subject with a precomputed BEM solution (5120-5120-5120-bem-sol.fif) 
    and a head->MRI coregistration transform (sample_audvis_raw-trans.fif).
    """
    data_path = Path(mne.datasets.sample.data_path(verbose=False))
    subjects_dir = data_path / "subjects"
    trans_path = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
    bem_path = data_path / "subjects" / "sample" / "bem" / "sample-5120-5120-5120-bem-sol.fif"

    if not bem_path.exists():
        raise FileNotFoundError(f"BEM solution not found at {bem_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Trans file not found at {trans_path}")

    print("  Loading precomputed BEM solution (3-shell, ico=4)...")
    bem_sol = mne.read_bem_solution(str(bem_path), verbose=False)

    prepare_mne_anatomy(
        anatomy_id="mne_sample",
        subject="sample",
        subjects_dir=subjects_dir,
        bem_sol=bem_sol,
        trans=trans_path,
        spacing=spacing,
        montage_ids=montage_ids,
        parcellation_schemes=parcellation_schemes,
        conductivity_id=conductivity_id,
        anatomy_bank_dir=anatomy_bank_dir,
        leadfield_bank_dir=leadfield_bank_dir,
        deepsif_anatomy_dir=deepsif_anatomy_dir,
    )

def _load_nyhead_mat(nyhead_path: Path) -> dict:
    """Load sa_nyhead.mat MATLAB v7.3 (HDF5/h5py)"""
    import h5py

    def _read_group(grp, f) -> dict:
        result = {}
        for key in grp.keys():
            item = grp[key]
            if isinstance(item, h5py.Group):
                result[key] = _read_group(item, f)
            elif isinstance(item, h5py.Dataset):
                data = item[()]
                if data.dtype == object:
                    try:
                        data = np.array(
                            ["".join(chr(c) for c in f[ref][()].flat)
                             for ref in data.flat],
                            dtype=str,
                        )
                    except Exception:
                        pass
                result[key] = data
        return result

    with h5py.File(str(nyhead_path), "r") as f:
        print(f"[LOG] Loading with h5py.File")
        return _read_group(f["sa"], f)


def _nyhead_adjacency(tri: np.ndarray, N: int) -> sp.csr_matrix:
    """Build a sparse (N, N) adjacency from a triangular face list (F, 3)."""
    tri = tri.astype(np.int32)
    if tri.shape[0] == 3:
        tri = tri.T                      # MATLAB stores as (3, F)
    if int(tri.min()) == 1:
        tri = tri - 1                    # MATLAB 1-indexed -> 0-indexed
    i0, i1, i2 = tri[:, 0], tri[:, 1], tri[:, 2]
    rows = np.concatenate([i0, i1, i1, i2, i2, i0])
    cols = np.concatenate([i1, i0, i2, i1, i0, i2])
    data = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


def _nyhead_parse_labels(raw) -> list[str]:
    """Parse the electrode label field from sa_nyhead.mat into a list of strings."""
    
    arr = np.asarray(raw)
    if arr.dtype.kind in ("U", "S", "O"):
        return [str(x).strip() for x in arr.flat]
    if arr.ndim == 2:
        # char matrix: each row is a padded string
        return ["".join(chr(int(c)) for c in row).strip() for row in arr]
    return [str(x).strip() for x in arr.flat]


def prepare_nyhead(
    anatomy_bank_dir: Path,
    leadfield_bank_dir: Path,
    montage_ids: list[str],
    conductivity_id: str = "standard",
    nyhead_path: Path | None = None,
) -> None:
    """Prepare anatomy + leadfield banks from the NY Head model.

    The New York Head (Huang et al., Brain Topography 2016) available at https://www.parralab.org/nyhead/
    """
    if nyhead_path is None or not nyhead_path.exists():
        raise FileNotFoundError(f"NY Head .mat file not found. Got: {nyhead_path}")

    print(f"  Loading sa_nyhead.mat from {nyhead_path}...")
    sa = _load_nyhead_mat(nyhead_path)

    cortex_key, lf_key = "cortex75K", "V_fem_normal"
    print(f"  Using cortex mesh: sa.{cortex_key}, leadfield: '{lf_key}'")
    
    vc_raw  = np.asarray(sa[cortex_key]["vc"],  dtype=np.float64)
    tri_raw = np.asarray(sa[cortex_key]["tri"])

    if vc_raw.shape[0] == 3 and vc_raw.shape[1] != 3:
        vc_raw = vc_raw.T  # MATLAB (3, N) → (N, 3)

    N = vc_raw.shape[0]
    scale = 1000.0 if np.abs(vc_raw).max() < 1.0 else 1.0
    vertex_coords = (vc_raw * scale).astype(np.float32)

    adjacency = _nyhead_adjacency(tri_raw, N)

    # Hemisphere: if x < 0 -> LH (0), if x > 0 -> RH (1) in MNI space
    hemisphere = (vertex_coords[:, 0] > 0).astype(np.int32)

    anatomy_dir = anatomy_bank_dir / "nyhead"
    anatomy_dir.mkdir(parents=True, exist_ok=True)
    (anatomy_dir / "parcellations").mkdir(parents=True, exist_ok=True)
    _save_source_space(
        anatomy_dir / "source_space.npz", vertex_coords, adjacency, hemisphere,
    )
    _save_parcellation(
        anatomy_dir / "parcellations" / "hemisphere.npz",
        hemisphere.astype(np.int32),
        ["left_hemisphere", "right_hemisphere"],
        scheme="hemisphere",
    )
    print(f"  Saved source space: {N} vertices")

    coord_raw = np.asarray(sa["locs_3D"], dtype=np.float32)
    labels_raw = sa["clab_electrodes"]
    
    coord_raw = coord_raw[:3, :].T   # from MATLAB, take xyz rows, shape (M, 3)
        
    elec_scale = 1000.0 if np.abs(coord_raw).max() < 1.0 else 1.0 # Unit detection
    elec_coords = (coord_raw * elec_scale).astype(np.float32)  # (M, 3) in mm
    M = elec_coords.shape[0]
    
    elec_labels = _nyhead_parse_labels(labels_raw)
    if len(elec_labels) != M:
        elec_labels = (elec_labels + [""] * M)[:M]

    lf_raw = np.asarray(sa[cortex_key][lf_key], dtype=np.float64)
    print(f"  Loaded leadfield '{lf_key}': raw shape {lf_raw.shape}")

    if lf_raw.ndim == 2:
        if lf_raw.shape[1] == M and lf_raw.shape[0] != M:
            lf_raw = lf_raw.T
        G_full = lf_raw
    else:
        raise ValueError(f"Unexpected fixed-lf shape: {lf_raw.shape}")

    G_full = G_full.astype(np.float32)
    print(f"  Leadfield ready: {G_full.shape}  ({M} electrodes x {N} sources)")

    # Build uppercase label -> electrode index for fast lookup
    nyhead_label_to_idx = {lbl.upper(): i for i, lbl in enumerate(elec_labels)}

    for montage_id in montage_ids:
        if montage_id not in MONTAGE_MAP:
            print(f"  Skipping unknown montage: {montage_id}")
            continue
        _extract_nyhead_montage(
            montage_id, elec_coords, nyhead_label_to_idx,
            G_full, leadfield_bank_dir, conductivity_id,
        )


def _extract_nyhead_montage(
    montage_id: str,
    nyhead_coords: np.ndarray,       # (M, 3) mm - all NY Head electrodes
    label_to_idx: dict[str, int],    # uppercase label -> row index in G_full
    G_full: np.ndarray,              # (M, N) full NY Head leadfield
    leadfield_bank_dir: Path,
    conductivity_id: str = "standard",
) -> None:
    """Extract a standard-montage subset from the NY Head full electrode set
    and write it as a single leadfield bundle.

    Channel matching strategy (in order of priority):
      1. Exact label match (case-insensitive).
      2. Nearest-neighbour in 3-D between the standard montage's positions
         (mm) and the NY Head electrode positions. A warning is printed for
         each fallback.

    The bundle uses the standard montage's channel names (not the NY Head
    labels), so it is compatible with the rest of the pipeline.
    """
    mne_name, n_channels = MONTAGE_MAP[montage_id]
    raw_montage = mne.channels.make_standard_montage(mne_name)
    positions = raw_montage.get_positions()["ch_pos"]
    ch_names = (
        _select_distributed_channels(positions, n_channels)
        if n_channels is not None and len(positions) > n_channels
        else list(positions.keys())
    )

    selected_indices: list[int] = []
    proximity_matched: list[str] = []
    nyhead_arr = nyhead_coords.astype(np.float32)

    for ch in ch_names:
        idx = label_to_idx.get(ch.upper())
        if idx is not None:
            selected_indices.append(idx)
        else:
            ch_mm = np.array(positions[ch], dtype=np.float32) * 1000.0
            dists = np.linalg.norm(nyhead_arr - ch_mm, axis=1)
            selected_indices.append(int(np.argmin(dists)))
            proximity_matched.append(ch)

    if proximity_matched:
        print(
            f"    {montage_id}: {len(proximity_matched)} channel(s) matched by "
            f"proximity (no exact label match): {proximity_matched}"
        )

    sel = np.array(selected_indices, dtype=np.int32)
    G = G_full[sel, :].astype(np.float32)
    coords_mm = nyhead_arr[sel]

    lf_path = leadfield_bank_dir / "nyhead" / montage_id / f"{conductivity_id}.npz"
    _save_leadfield(lf_path, G, ch_names, coords_mm)
    print(f"    Saved leadfield nyhead/{montage_id}/{conductivity_id}: {G.shape}")


# Registry and CLI

_PREPARE_FNS: dict[str, Callable[..., None]] = {
    "fsaverage":  prepare_fsaverage,
    "mne_sample": prepare_mne_sample,
    "nyhead":     prepare_nyhead,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare synthgen anatomy, leadfield, and montage bank files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to GenerationConfig YAML file")
    parser.add_argument(
        "--anatomy", nargs="+", default=None,
        help="Anatomy IDs to prepare (default: all listed in config)",
    )
    parser.add_argument(
        "--spacing", default="ico5",
        help="MNE source-space spacing. ico5 = fsaverage5 (20484 vertices, matches "
             "DeepSIF). Alternatives: oct6 (~8k), ico4 (~5k), oct7 (~32k). "
             "Default: ico5.",
    )
    parser.add_argument(
        "--parcellation", nargs="+", default=["deepsif_994"],
        help="Parcellation scheme(s) to compute and store under "
             "parcellations/{scheme}.npz. One or more of: desikan_killiany, "
             "destrieux, hcp_mmp1, deepsif_994, schaefer_400, schaefer_1000.",
    )
    parser.add_argument(
        "--deepsif-dir", type=Path, default=None, dest="deepsif_dir",
        help="Path to the DeepSIF anatomy directory (containing "
             "fs_cortex_20k_region_mapping.mat). Required only when "
             "'deepsif_994' is in --parcellation. Fetch via "
             "scripts/fetch_atlases.py --deepsif.",
    )
    parser.add_argument(
        "--nyhead-mat", type=Path, default=None, dest="nyhead_mat",
        help="Path to sa_nyhead.mat (required when preparing 'nyhead' anatomy). "
             "Download from https://www.parralab.org/nyhead/",
    )
    parser.add_argument(
        "--conductivity-id", type=str, default="standard",
        help="Conductivity tag stored in the leadfield filename (default: standard).",
    )
    args = parser.parse_args()

    config = GenerationConfig.from_yaml(args.config)
    anatomy_bank_dir  = Path(config.anatomy_bank.bank_dir)
    leadfield_bank_dir = Path(config.leadfield_bank.bank_dir)
    anatomy_ids = args.anatomy or config.anatomy_bank.anatomy_ids
    montage_ids = [m.name for m in config.montages.montages]

    for anatomy_id in anatomy_ids:
        print(f"\nPreparing anatomy: {anatomy_id}")
        fn = _PREPARE_FNS.get(anatomy_id)
        if fn is None:
            print(f"  Unknown anatomy '{anatomy_id}', skipping.")
            continue
        if anatomy_id == "nyhead":
            fn(anatomy_bank_dir, leadfield_bank_dir,
               montage_ids, args.conductivity_id, args.nyhead_mat)
        else:
            fn(anatomy_bank_dir, leadfield_bank_dir,
               montage_ids, args.spacing, args.parcellation,
               args.conductivity_id, args.deepsif_dir)


if __name__ == "__main__":
    main()