#!/usr/bin/env python3
"""Fetch external parcellation atlases not shipped with MNE or tvb-data.

Usage:
    # DeepSIF 994 regions (Sun et al. 2022) + 998-region Hagmann-style connectome
    python scripts/fetch_atlases.py --deepsif

    # Schaefer 2018 functional parcellations at selected resolutions
    python scripts/fetch_atlases.py --schaefer --n-parcels 400 1000

    # Combine
    python scripts/fetch_atlases.py --deepsif --schaefer --n-parcels 400 1000

Outputs
-------
DeepSIF:
    banks/atlases/deepsif/
        fs_cortex_20k_region_mapping.mat   (20484 -> 994 region ids)
        connectivity_998.zip               (TVB-compatible connectome)

Schaefer:
    $FREESURFER_SUBJECTS_DIR/fsaverage/label/
        {lh,rh}.Schaefer2018_<N>Parcels_17Networks_order.annot
    Or, if FREESURFER_SUBJECTS_DIR is unset, the MNE fsaverage path is used.
"""
from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

DEEPSIF_RAW = "https://raw.githubusercontent.com/bfinl/DeepSIF/main/anatomy/{name}"
DEEPSIF_FILES = ("fs_cortex_20k_region_mapping.mat", "connectivity_998.zip")

SCHAEFER_RAW = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage/label/"
    "{hemi}.Schaefer2018_{n}Parcels_17Networks_order.annot"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  [get ] {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"         -> {dest} ({dest.stat().st_size / 1024:.1f} kB)")


def fetch_deepsif(out_dir: Path) -> None:
    print(f"Fetching DeepSIF atlas into {out_dir}")
    for name in DEEPSIF_FILES:
        _download(DEEPSIF_RAW.format(name=name), out_dir / name)


def _fsaverage_label_dir() -> Path:
    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if subjects_dir:
        p = Path(subjects_dir) / "fsaverage" / "label"
        if p.exists():
            return p
    import mne
    return Path(mne.datasets.fetch_fsaverage(verbose=False)) / "label"


def fetch_schaefer(n_parcels_list: list[int]) -> None:
    label_dir = _fsaverage_label_dir()
    print(f"Fetching Schaefer atlas(es) into {label_dir}")
    for n in n_parcels_list:
        for hemi in ("lh", "rh"):
            url = SCHAEFER_RAW.format(hemi=hemi, n=n)
            dest = label_dir / f"{hemi}.Schaefer2018_{n}Parcels_17Networks_order.annot"
            _download(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deepsif", action="store_true", help="Fetch DeepSIF-994 region mapping + 998 connectome.")
    ap.add_argument("--deepsif-out", type=Path, default=Path("banks/atlases/deepsif"),
                    help="Output directory for DeepSIF files.")
    ap.add_argument("--schaefer", action="store_true", help="Fetch Schaefer 2018 .annot files.")
    ap.add_argument("--n-parcels", type=int, nargs="+", default=[400, 1000],
                    help="Schaefer parcel counts to fetch (supported: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000).")
    args = ap.parse_args()

    if not args.deepsif and not args.schaefer:
        ap.error("Nothing to do: pass --deepsif and/or --schaefer.")

    if args.deepsif:
        fetch_deepsif(args.deepsif_out)
    if args.schaefer:
        fetch_schaefer(args.n_parcels)


if __name__ == "__main__":
    main()
