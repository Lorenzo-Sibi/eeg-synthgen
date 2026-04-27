#!/usr/bin/env python3
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


def _save_leadfield(path: Path, G: np.ndarray) -> None:
    np.savez_compressed(path, G=G.astype(np.float32))


def _save_montage(path: Path, coords: np.ndarray, ch_names: list[str]) -> None:
    np.savez_compressed(path, coords=coords.astype(np.float32), ch_names=np.array(ch_names, dtype=str))


def _select_distributed_channels(
    ch_pos: dict[str, np.ndarray],
    n: int,
) -> list[str]:
    """Select n channels from ch_pos with maximum scalp coverage.

    Uses farthest-point (maximin) sampling: seed at the electrode closest to
    the crown of the head (highest z in MNE head frame), then iteratively
    add the electrode that maximises the minimum distance to all already-selected
    electrodes.  The result is a well-distributed, scalp-covering subset for any
    target count n, with no domain knowledge required.

    Time complexity: O(C * n) — negligible for typical montage sizes.
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
    src: list,
    inuse_src: list | None = None,
    mri_head_t: Any | None = None,
) -> tuple[np.ndarray, sp.spmatrix, np.ndarray]:
    """Extract geometry (vertex_coords mm, adjacency, hemisphere) from MNE src.

    Parameters
    ----------
    src :
        Source space used for vertex positions (rr). Must be in MRI frame (the
        original src returned by setup_source_space, coord_frame=5).
    inuse_src :
        If provided, use its ``inuse`` mask and adjacency instead of src's. Pass
        fwd["src"] here because MNE may silently drop vertices outside the BEM
        surface, so fwd["src"]["inuse"] is authoritative for N. After
        make_forward_solution, fwd["src"]["coord_frame"] = HEAD (4), so do NOT
        use fwd["src"]["rr"] for positions — always pass the original src.

    Returns
    -------
    vertex_coords : (N, 3) float32, mm, head frame if ``mri_head_t`` is given.
    adjacency :    (N, N) sparse — cortical neighbourhood graph from MNE.
    hemisphere :   (N,)   int32  — 0=left hemisphere, 1=right hemisphere.
    """
    mask_src = inuse_src if inuse_src is not None else src
    adjacency = mne.spatial_src_adjacency(mask_src, verbose=False)

    lh_used = mask_src[0]["inuse"].astype(bool)
    rh_used = mask_src[1]["inuse"].astype(bool)
    # src["rr"] stores vertex coordinates in metres in MRI frame -> convert to mm
    lh_coords = src[0]["rr"][lh_used] * 1000.0
    rh_coords = src[1]["rr"][rh_used] * 1000.0
    vertex_coords = np.concatenate([lh_coords, rh_coords], axis=0).astype(np.float32)

    if mri_head_t is not None:
        vertex_coords = (apply_trans(mri_head_t, vertex_coords / 1000.0) * 1000.0).astype(np.float32)

    n_lh = int(lh_used.sum())
    n_rh = int(rh_used.sum())
    hemisphere = np.concatenate(
        [np.zeros(n_lh, dtype=np.int32), np.ones(n_rh, dtype=np.int32)]
    )
    return vertex_coords, adjacency, hemisphere


def _prepare_montage_leadfield(
    montage_id: str,
    src: list,
    bem_sol: Any,
    montage_bank_dir: Path,
    leadfield_bank_dir: Path,
    anatomy_id: str,
    trans: str | Path = "fsaverage",
) -> tuple[list, Any]:
    """Compute and save the EEG forward solution (leadfield) for one montage.

    The leadfield G has shape (C, N), where C = number of channels and N = number
    of cortical source vertices. Dipoles are oriented perpendicular to the cortex
    (force_fixed=True, surf_ori=True).

    Returns (fwd["src"], fwd["mri_head_t"]).
    fwd["src"] is the source space actually used in the forward computation — MNE may
    silently drop vertices outside the BEM surface, so it is authoritative for N.
    fwd["mri_head_t"] is the MNE Transform object mapping MRI coordinates to HEAD
    coordinates; callers use it to store vertex_coords in head frame.

    Parameters
    ----------
    trans :
        Transformation from head coordinates to MRI coordinates.
        - "fsaverage": MNE built-in trans for the fsaverage subject.
        - Path to a subject-specific *-trans.fif file (e.g. from MEG coregistration).
    """

    mne_name, n_channels = MONTAGE_MAP[montage_id]
    raw_montage = mne.channels.make_standard_montage(mne_name)
    positions = raw_montage.get_positions()
    ch_pos = positions["ch_pos"]
    # Fiducials: including these eliminates the "nasion not found" RuntimeWarning
    # and ensures the montage is properly anchored in the MNE head coordinate frame.
    nasion = positions.get("nasion")
    lpa    = positions.get("lpa")
    rpa    = positions.get("rpa")

    ch_names = (
        _select_distributed_channels(ch_pos, n_channels)
        if n_channels is not None and len(ch_pos) > n_channels
        else list(ch_pos.keys())
    )

    ch_pos_subset = {ch: ch_pos[ch] for ch in ch_names}
    coords_mm = np.array([ch_pos[ch] for ch in ch_names], dtype=np.float32) * 1000.0 # metres to mm

    montage_bank_dir.mkdir(parents=True, exist_ok=True)
    montage_path = montage_bank_dir / f"{montage_id}.npz"
    if not montage_path.exists():
        _save_montage(montage_path, coords_mm, ch_names)
        print(f"    Saved montage {montage_id}: {len(ch_names)} channels")

    # Build info with digitised montage (fiducials included -> no warning)
    info = mne.create_info(ch_names, sfreq=500, ch_types="eeg")
    dig_kwargs: dict[str, Any] = {"ch_pos": ch_pos_subset}
    if nasion is not None:
        dig_kwargs["nasion"] = nasion
    if lpa is not None:
        dig_kwargs["lpa"] = lpa
    if rpa is not None:
        dig_kwargs["rpa"] = rpa
    dig_montage = mne.channels.make_dig_montage(**dig_kwargs)
    info.set_montage(dig_montage, on_missing="ignore", verbose=False)

    print(f"    Computing forward solution ({montage_id}, {len(ch_names)} ch)...")
    fwd = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem_sol,
        meg=False, eeg=True, n_jobs=1, verbose=False,
    )
    # surf_ori=True  -> dipole z-axis aligned to cortical surface normal
    # force_fixed=True -> project onto normal, giving (C, N) instead of (C, 3N)
    fwd_fixed = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, verbose=False
    )
    G = fwd_fixed["sol"]["data"]  # Leadfield (C, N)
    lf_dir = leadfield_bank_dir / anatomy_id / montage_id
    lf_dir.mkdir(parents=True, exist_ok=True)
    
    _save_leadfield(lf_dir / "standard.npz", G)
    print(f"    Saved leadfield {anatomy_id}/{montage_id}/standard: {G.shape}")

    elec_coords_mm = np.array(
        [ch["loc"][:3] for ch in fwd["info"]["chs"]], dtype=np.float32
    ) * 1000.0
    _save_montage(lf_dir / "electrode_coords.npz", elec_coords_mm, ch_names)
    
    return fwd["src"], fwd["mri_head_t"]

def _run_mne_pipeline(
    anatomy_id : str,
    subject: str,
    subjects_dir: str,
    bem_solution: Any,
    trans: str | Path,
    spacing: str,
    montage_ids: list[str],
    parcellation_schemes: list[str],
    anatomy_bank_dir: Path,
    leadfield_bank_dir: Path,
    montage_bank_dir: Path,
    deepsif_anatomy_dir: Path | None = None,
) -> None:
    """Shared pipeline for MNE-based anatomies.

    For each scheme in ``parcellation_schemes`` a file is written at
    ``anatomy_bank_dir/{anatomy_id}/parcellations/{scheme}.npz``.
    """

    print(f"  Setting up source space (spacing={spacing})...")
    src = mne.setup_source_space(
        subject, spacing=spacing, subjects_dir=subjects_dir,
        add_dist="patch", verbose=False,
    )

    anatomy_dir = anatomy_bank_dir / anatomy_id
    anatomy_dir.mkdir(parents=True, exist_ok=True)
    (anatomy_dir / "parcellations").mkdir(parents=True, exist_ok=True)

    fwd_src, mri_head_t = None, None
    for montage_id in montage_ids:
        if montage_id not in MONTAGE_MAP:
            print(f"  Skipping unknown montage: {montage_id}")
            continue
        fwd_src, mri_head_t = _prepare_montage_leadfield(
            montage_id, src, bem_solution, montage_bank_dir, leadfield_bank_dir,
            anatomy_id, trans=trans,
        )

    vertex_coords, adjacency, hemisphere = _extract_source_space(
        src, inuse_src=fwd_src, mri_head_t=None,
    )

    _save_source_space(
        anatomy_dir / "source_space.npz", vertex_coords, adjacency, hemisphere,
    )
    print(f"  Saved source space: {len(vertex_coords)} vertices")

    for scheme in parcellation_schemes:
        try:
            parcellation, label_names = _read_parcellation(
                scheme, subject, subjects_dir, src,
                deepsif_anatomy_dir=deepsif_anatomy_dir,
                inuse_src=fwd_src,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"    Skipping parcellation {scheme!r}: {e}")
            continue
        _save_parcellation(
            anatomy_dir / "parcellations" / f"{scheme}.npz",
            parcellation, label_names, scheme,
        )
        print(f"    Saved parcellation {scheme}: {len(label_names)} regions")

def prepare_fsaverage(
    anatomy_bank_dir: Path,
    leadfield_bank_dir: Path,
    montage_bank_dir: Path,
    montage_ids: list[str],
    spacing: str,
    parcellation_schemes: list[str],
    deepsif_anatomy_dir: Path | None = None,
) -> None:
    """Prepare anatomy + leadfield banks for the MNE fsaverage subject.

    fsaverage is the FreeSurfer average brain.  MNE ships a pre-built
    trans ("fsaverage") that aligns the standard 10-05 montage head frame
    to the fsaverage MRI frame, so no separate coregistration is needed.

    BEM model choices
    -----------------
    ico=4 : ico=4 120 triangles per BEM surface - standard accuracy/cost trade-off;
            ico=3 (1 280 tri) introduces visible numerical errors; ico=5 (20 480)
            improves localisation by <0.5 mm. Default set to ico=4, adjust accordingly.
    conductivity=[0.3, 0.006, 0.3] S/m : brain, skull, scalp.
            The skull value (0.006) gives a brain/skull ratio of 1/50 (Geddes &
            Baker 1967), the classical EEG value.  Alternatives (0.004-0.015 S/m)
            change leadfield amplitudes by ~5-15 %, acceptable for this use.
    3-shell : adds CSF layer in 4-shell models -> <10 % improvement at the cost
            of explicit CSF segmentation.  3-shell is the MNE default and is
            appropriate for training data diversity.
    """
    subjects_dir = Path(mne.datasets.fetch_fsaverage(verbose=False)).parent
    bem_model = mne.make_bem_model(
        "fsaverage", ico=4, conductivity=[0.3, 0.006, 0.3],
        subjects_dir=subjects_dir, verbose=False,
    )
    print("  Building BEM model (3-shell, ico=4)...")
    bem_sol = mne.make_bem_solution(bem_model, verbose=False)

    _run_mne_pipeline(
        anatomy_id="fsaverage",
        subject="fsaverage",
        subjects_dir=subjects_dir,
        bem_solution=bem_sol,
        trans="fsaverage",
        spacing=spacing,
        montage_ids=montage_ids,
        parcellation_schemes=parcellation_schemes,
        anatomy_bank_dir=anatomy_bank_dir,
        leadfield_bank_dir=leadfield_bank_dir,
        montage_bank_dir=montage_bank_dir,
        deepsif_anatomy_dir=deepsif_anatomy_dir,
    )


def prepare_mne_sample(
    anatomy_bank_dir: Path,
    leadfield_bank_dir: Path,
    montage_bank_dir: Path,
    montage_ids: list[str],
    spacing: str,
    parcellation_schemes: list[str],
    deepsif_anatomy_dir: Path | None = None,
) -> None:
    """Prepare anatomy + leadfield banks for the MNE Sample subject.

    The MNE sample dataset includes a real subject with:
      - FreeSurfer cortical surface reconstruction
      - A precomputed 3-shell BEM solution (5120-5120-5120-bem-sol.fif)
      - A head-to-MRI coregistration transform (sample_audvis_raw-trans.fif)

    Trans file note
    ---------------
    sample_audvis_raw-trans.fif encodes the head->MRI transformation derived
    from coregistering the subject's digitised nasion/LPA/RPA (Polhemus) to
    their MRI.  When a *standard* montage (with idealised fiducials) is used
    instead of the actual digitised positions, a small misalignment of ~2-5 mm
    is introduced.  This is acceptable for generating diverse training data.
    """

    data_path = Path(mne.datasets.sample.data_path(verbose=False))
    subjects_dir = data_path / "subjects"
    trans_path = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
    bem_path = data_path / "subjects" / "sample" / "bem" / "sample-5120-5120-5120-bem-sol.fif"

    if not bem_path.exists() or not trans_path.exists():
        raise FileNotFoundError(
            f"BEM solution not found at {bem_path}. " if not bem_path.exists() else "" +
            f"Trans file not found at {trans_path}. " if not trans_path.exists() else "" +
            "Make sure the MNE sample dataset is fully downloaded."
        )
    print("  Loading precomputed BEM solution (3-shell, ico=4)...")
    bem_sol = mne.read_bem_solution(str(bem_path), verbose=False) # The precomputed BEM uses ico=4 surfaces (5120 tri/surface)

    _run_mne_pipeline(
        anatomy_id="mne_sample",
        subject="sample",
        subjects_dir=subjects_dir,
        bem_solution=bem_sol,
        trans=trans_path,
        spacing=spacing,
        montage_ids=montage_ids,
        parcellation_schemes=parcellation_schemes,
        anatomy_bank_dir=anatomy_bank_dir,
        leadfield_bank_dir=leadfield_bank_dir,
        montage_bank_dir=montage_bank_dir,
        deepsif_anatomy_dir=deepsif_anatomy_dir,
    )

def _load_nyhead_mat(nyhead_path: Path) -> dict:
    """Load sa_nyhead.mat MATLAB v7.3 (HDF5/h5py).
    Returns the 'sa' struct as a nested Python dict of numpy arrays.
    MATLAB v7.3 HDF5 files store strings as arrays of uint16 Unicode code points.
    Struct fields become HDF5 Groups; arrays become Datasets.
    """
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
    montage_bank_dir: Path,
    montage_ids: list[str],
    nyhead_path: Path | None = None,
) -> None:
    """Prepare anatomy + leadfield banks from the NY Head model.

    The New York Head (Huang et al., Brain Topography 2016) available at
    https://www.parralab.org/nyhead/
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
            G_full, montage_bank_dir, leadfield_bank_dir,
        )


def _extract_nyhead_montage(
    montage_id: str,
    nyhead_coords: np.ndarray,       # (M, 3) mm - all NY Head electrodes
    label_to_idx: dict[str, int],    # uppercase label -> row index in G_full
    G_full: np.ndarray,              # (M, N) full NY Head leadfield
    montage_bank_dir: Path,
    leadfield_bank_dir: Path,
) -> None:
    """Extract a standard-montage subset from the NY Head full electrode set.

    Channel matching strategy (in order of priority):
    1. Exact label match (case-insensitive).
    2. Nearest-neighbour in 3-D: standard montage position (converted to mm)
       vs. NY Head electrode position.  A warning is printed for each fallback.

    The extracted montage bank file uses the standard montage's channel names
    (not the NY Head labels), so it is compatible with the rest of the pipeline.
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
            # Nearest-neighbour fallback (standard_1005 pos in metres -> mm)
            ch_mm = np.array(positions[ch], dtype=np.float32) * 1000.0
            dists = np.linalg.norm(nyhead_arr - ch_mm, axis=1)
            nn_idx = int(np.argmin(dists))
            selected_indices.append(nn_idx)
            proximity_matched.append(ch)

    if proximity_matched:
        print(
            f"    {montage_id}: {len(proximity_matched)} channel(s) matched by "
            f"proximity (no exact label match): {proximity_matched}"
        )

    sel = np.array(selected_indices, dtype=np.int32)
    G = G_full[sel, :]                    # (C, N)
    coords_mm = nyhead_arr[sel]           # (C, 3)

    montage_bank_dir.mkdir(parents=True, exist_ok=True)
    montage_path = montage_bank_dir / f"{montage_id}.npz"
    if not montage_path.exists():
        _save_montage(montage_path, coords_mm, ch_names)
        print(f"    Saved montage {montage_id}: {len(ch_names)} channels")

    lf_dir = leadfield_bank_dir / "nyhead" / montage_id
    lf_dir.mkdir(parents=True, exist_ok=True)
    _save_leadfield(lf_dir / "standard.npz", G)
    _save_montage(lf_dir / "electrode_coords.npz", coords_mm, ch_names)
    print(f"    Saved leadfield nyhead/{montage_id}/standard: {G.shape}")


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
    args = parser.parse_args()

    config = GenerationConfig.from_yaml(args.config)
    anatomy_bank_dir  = Path(config.anatomy_bank.bank_dir)
    leadfield_bank_dir = Path(config.leadfield_bank.bank_dir)
    montage_bank_dir  = Path(config.montage_bank.bank_dir)
    anatomy_ids = args.anatomy or config.anatomy_bank.anatomy_ids
    montage_ids = [m.name for m in config.montage_bank.montages]

    for anatomy_id in anatomy_ids:
        print(f"\nPreparing anatomy: {anatomy_id}")
        fn = _PREPARE_FNS.get(anatomy_id)
        if fn is None:
            print(f"  Unknown anatomy '{anatomy_id}', skipping.")
            continue
        if anatomy_id == "nyhead":
            fn(anatomy_bank_dir, leadfield_bank_dir, montage_bank_dir,
               montage_ids, args.nyhead_mat)
        else:
            fn(anatomy_bank_dir, leadfield_bank_dir, montage_bank_dir,
               montage_ids, args.spacing, args.parcellation, args.deepsif_dir)


if __name__ == "__main__":
    main()