"""
One-shot diagnostic for EEG bank files.

Usage:
    python scripts/validate_banks.py [--banks-dir banks/] [--plot]

Exit codes: 0 = all PASS/WARN, 1 = any FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def _load_anatomy_raw(anat_dir: Path):
    """Load anatomy geometry from source_space.npz and parcellations (if present).

    Returns (vertex_coords, adjacency, parcellations_by_scheme, hemisphere) where
    parcellations_by_scheme is a dict {scheme_name: parcellation_array}. Supports
    both the new split layout (parcellations/{scheme}.npz) and legacy source_space
    files with an embedded 'parcellation' key.
    """
    data = np.load(anat_dir / "source_space.npz", allow_pickle=False)
    adj = sp.csr_matrix(
        (data["adjacency_data"], data["adjacency_indices"], data["adjacency_indptr"]),
        shape=tuple(data["adjacency_shape"]),
    )
    parcellations: dict[str, np.ndarray] = {}
    parc_dir = anat_dir / "parcellations"
    if parc_dir.exists():
        for p in sorted(parc_dir.glob("*.npz")):
            parcellations[p.stem] = np.load(p, allow_pickle=False)["parcellation"]
    if not parcellations and "parcellation" in data.files:
        scheme = (
            str(data["parcellation_scheme"])
            if "parcellation_scheme" in data.files
            else "legacy"
        )
        parcellations[scheme] = data["parcellation"]
    return data["vertex_coords"], adj, parcellations, data["hemisphere"]


def _load_leadfield_raw(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)["G"]


def _load_montage_raw(path: Path):
    data = np.load(path, allow_pickle=False)
    return data["coords"], [str(s) for s in data["ch_names"]]


def check_anatomies(banks_dir: Path):
    results, fails = [], 0
    anat_dir = banks_dir / "anatomy"
    if not anat_dir.exists():
        return results, fails
    for anat_path in sorted(p for p in anat_dir.iterdir() if p.is_dir()):
        ss_path = anat_path / "source_space.npz"
        if not ss_path.exists():
            results.append({"name": anat_path.name, "status": "FAIL",
                            "detail": "source_space.npz missing"})
            fails += 1
            continue
        try:
            vc, adj, parcellations, hemisphere = _load_anatomy_raw(anat_path)
        except Exception as exc:
            results.append({"name": anat_path.name, "status": "FAIL", "detail": str(exc)})
            fails += 1
            continue

        N = vc.shape[0]
        issues = []
        if vc.shape != (N, 3):
            issues.append(f"vertex_coords shape {vc.shape}")
        if vc.dtype != np.float32:
            issues.append(f"vertex_coords dtype {vc.dtype}")
        if np.any(np.isnan(vc)) or np.any(np.isinf(vc)):
            issues.append("NaN/Inf in vertex_coords")
        if vc.min() < -150 or vc.max() > 150:
            issues.append(f"coords out of [-150,150] mm: [{vc.min():.0f},{vc.max():.0f}]")
        if not set(np.unique(hemisphere).tolist()).issubset({0, 1}):
            issues.append("hemisphere values not in {0,1}")
        n_lh = int((hemisphere == 0).sum())
        n_rh = int((hemisphere == 1).sum())
        if n_lh == 0 or n_rh == 0:
            issues.append(f"missing hemisphere L={n_lh} R={n_rh}")
        diff = adj - adj.T
        diff.eliminate_zeros()
        if diff.nnz != 0:
            issues.append("adjacency not symmetric")
        for scheme, parc in parcellations.items():
            if parc.shape[0] != N:
                issues.append(f"parcellation/{scheme} len {parc.shape[0]} != N={N}")

        if issues:
            fails += 1
        x, y, z = vc[:, 0], vc[:, 1], vc[:, 2]
        if parcellations:
            parcel_str = ", ".join(
                f"{scheme}:{len(np.unique(p))}" for scheme, p in parcellations.items()
            )
        else:
            parcel_str = "none"
        results.append({
            "name": anat_path.name, "N": N,
            "coord_range": (f"x[{x.min():.0f},{x.max():.0f}] "
                            f"y[{y.min():.0f},{y.max():.0f}] "
                            f"z[{z.min():.0f},{z.max():.0f}]"),
            "hemi": f"L={n_lh} R={n_rh}",
            "parcels": parcel_str,
            "status": "FAIL" if issues else "PASS",
            "detail": "; ".join(issues),
        })
    return results, fails


def check_montages(banks_dir: Path):
    results, fails = [], 0
    mont_dir = banks_dir / "montage"
    if not mont_dir.exists():
        return results, fails
    for npz in sorted(mont_dir.glob("*.npz")):
        try:
            coords, ch_names = _load_montage_raw(npz)
        except Exception as exc:
            results.append({"name": npz.stem, "status": "FAIL", "detail": str(exc)})
            fails += 1
            continue
        C = len(ch_names)
        issues = []
        if coords.shape != (C, 3):
            issues.append(f"coords shape {coords.shape} != ({C},3)")
        if coords.dtype != np.float32:
            issues.append(f"coords dtype {coords.dtype}")
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            issues.append("NaN/Inf in coords")
        if coords.min() < -150 or coords.max() > 150:
            issues.append("coords out of [-150,150] mm")
        if issues:
            fails += 1
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        results.append({
            "name": npz.stem, "C": C,
            "coord_range": (f"x[{x.min():.0f},{x.max():.0f}] "
                            f"y[{y.min():.0f},{y.max():.0f}] "
                            f"z[{z.min():.0f},{z.max():.0f}]"),
            "status": "FAIL" if issues else "PASS",
            "detail": "; ".join(issues),
        })
    return results, fails


def check_leadfields(banks_dir: Path):
    results, fails = [], 0
    lf_dir = banks_dir / "leadfield"
    if not lf_dir.exists():
        return results, fails
    for anat_d in sorted(p for p in lf_dir.iterdir() if p.is_dir()):
        for mont_d in sorted(p for p in anat_d.iterdir() if p.is_dir()):
            for npz in sorted(p for p in mont_d.glob("*.npz") if p.name != "electrode_coords.npz"):
                label = f"{anat_d.name} / {mont_d.name} / {npz.stem}"
                try:
                    G = _load_leadfield_raw(npz)
                except Exception as exc:
                    results.append({"name": label, "status": "FAIL", "detail": str(exc)})
                    fails += 1
                    continue
                issues = []
                if G.dtype != np.float32:
                    issues.append(f"dtype {G.dtype}")
                n_nan = int(np.sum(np.isnan(G)))
                n_inf = int(np.sum(np.isinf(G)))
                if n_nan:
                    issues.append(f"NaN={n_nan}")
                if n_inf:
                    issues.append(f"Inf={n_inf}")
                col_norms = np.linalg.norm(G, axis=0)
                n_zero = int((col_norms == 0).sum())
                if n_zero:
                    issues.append(f"zero_cols={n_zero}")
                ratio = col_norms.max() / (col_norms.mean() + 1e-30)
                if ratio > 1e4:
                    issues.append(f"max/mean ratio {ratio:.1e}")
                cv = col_norms.std() / (col_norms.mean() + 1e-30)
                if cv < 0.05:
                    issues.append(f"CV={cv:.3f} (spatially uniform?)")
                if issues:
                    fails += 1
                results.append({
                    "name": label,
                    "shape": f"({G.shape[0]},{G.shape[1]})",
                    "col_norm": (f"[min={col_norms.min():.2e} "
                                 f"mean={col_norms.mean():.2e} "
                                 f"max={col_norms.max():.2e}]"),
                    "NaN": n_nan,
                    "status": "FAIL" if issues else "PASS",
                    "detail": "; ".join(issues),
                })
    return results, fails


def check_coherence(banks_dir: Path):
    results, fails = [], 0
    lf_dir = banks_dir / "leadfield"
    if not lf_dir.exists():
        return results, fails
    for anat_d in sorted(p for p in lf_dir.iterdir() if p.is_dir()):
        anat_ss = banks_dir / "anatomy" / anat_d.name / "source_space.npz"
        for mont_d in sorted(p for p in anat_d.iterdir() if p.is_dir()):
            mont_npz = banks_dir / "montage" / f"{mont_d.name}.npz"
            for npz in sorted(p for p in mont_d.glob("*.npz") if p.name != "electrode_coords.npz"):
                label = f"{anat_d.name}__{mont_d.name}__{npz.stem}"
                issues = []
                try:
                    G = _load_leadfield_raw(npz)
                    C_G, N_G = G.shape
                except Exception as exc:
                    results.append({"name": label, "status": "FAIL",
                                    "detail": f"leadfield: {exc}"})
                    fails += 1
                    continue
                N_anat = C_mont = None
                if anat_ss.exists():
                    try:
                        d = np.load(anat_ss, allow_pickle=False)
                        N_anat = len(d["vertex_coords"])
                    except Exception as exc:
                        issues.append(f"anatomy load: {exc}")
                else:
                    issues.append("anatomy file missing")
                if mont_npz.exists():
                    try:
                        d = np.load(mont_npz, allow_pickle=False)
                        C_mont = len(d["ch_names"])
                    except Exception as exc:
                        issues.append(f"montage load: {exc}")
                else:
                    issues.append("montage file missing")
                if N_anat is not None and N_G != N_anat:
                    issues.append(f"N mismatch G={N_G} anatomy={N_anat}")
                if C_mont is not None and C_G != C_mont:
                    issues.append(f"C mismatch G={C_G} montage={C_mont}")
                if issues:
                    fails += 1
                results.append({
                    "name": label,
                    "c_match": f"C {C_G}=={C_mont}" if C_mont is not None else "C: montage missing",
                    "n_match": f"N {N_G}=={N_anat}" if N_anat is not None else "N: anatomy missing",
                    "status": "FAIL" if issues else "PASS",
                    "detail": "; ".join(issues),
                })
    return results, fails


def _try_mne_alignment(
    anat_id: str,
    mont_id: str,
    mc_mm: np.ndarray,
    ch_names: list[str],
) -> None:
    """Call mne.viz.plot_alignment for MNE-based anatomies (fsaverage / mne_sample).

    mc_mm : (C, 3) electrode coordinates in mm, anatomy head frame.
    Silently skipped for unknown anatomies or when MNE data is unavailable.
    """
    try:
        import mne
    except ImportError:
        return

    def _fsaverage_info():
        fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
        return "fsaverage", fs_dir.parent, str(fs_dir / "bem" / "fsaverage-trans.fif")

    def _sample_info():
        sd = mne.datasets.sample.data_path()
        return "sample", sd / "subjects", str(sd / "MEG" / "sample" / "sample_audvis_raw-trans.fif")

    _LOOKUP = {"fsaverage": _fsaverage_info, "mne_sample": _sample_info}
    if anat_id not in _LOOKUP:
        return

    try:
        subject, subjects_dir, trans = _LOOKUP[anat_id]()
    except Exception as exc:
        print(f"  [plot_alignment] {anat_id}: MNE data unavailable — {exc}")
        return

    try:
        info = mne.create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
        dig = mne.channels.make_dig_montage(
            ch_pos={name: pos / 1000.0 for name, pos in zip(ch_names, mc_mm)}
        )
        with mne.utils.use_log_level("warning"):
            info.set_montage(dig)
        print(f"  [plot_alignment] {anat_id}/{mont_id} — close window to continue")
        mne.viz.plot_alignment(
            info,
            trans=trans,
            subject=subject,
            subjects_dir=subjects_dir,
            surfaces=["white", "outer_skin"],
            coord_frame="head",
            show_axes=True,
        )
    except Exception as exc:
        print(f"  [plot_alignment] {anat_id}/{mont_id}: {exc}")


def _plot_banks(banks_dir: Path) -> None:
    import math
    import matplotlib.pyplot as plt

    anat_dir = banks_dir / "anatomy"
    lf_dir   = banks_dir / "leadfield"
    mont_dir = banks_dir / "montage"

    anatomy_ids = (
        [d.name for d in sorted(anat_dir.iterdir()) if d.is_dir()]
        if anat_dir.exists() else []
    )
    montage_ids = (
        [p.stem for p in sorted(mont_dir.glob("*.npz"))]
        if mont_dir.exists() else []
    )

    # ── 1. Source space + electrode scatter — one figure per anatomy ──────────
    for anat_id in anatomy_ids:
        ss_path = anat_dir / anat_id / "source_space.npz"
        if not ss_path.exists():
            continue
        ss = np.load(ss_path, allow_pickle=False)
        vc   = ss["vertex_coords"]
        hemi = ss["hemisphere"]

        shown_montages = [
            m for m in montage_ids
            if (lf_dir / anat_id / m / "electrode_coords.npz").exists()
               or (mont_dir / f"{m}.npz").exists()
        ]
        if not shown_montages:
            continue

        ncols = min(len(shown_montages), 4)
        nrows = math.ceil(len(shown_montages) / ncols)
        fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
        fig.suptitle(
            f"{anat_id} — source space (blue=LH, red=RH) + electrodes (gold)",
            fontsize=10,
        )

        for i, mont_id in enumerate(shown_montages):
            elec_npz  = lf_dir / anat_id / mont_id / "electrode_coords.npz"
            mont_path = mont_dir / f"{mont_id}.npz"
            src = elec_npz if elec_npz.exists() else mont_path if mont_path.exists() else None
            if src is None:
                continue
            d  = np.load(src, allow_pickle=False)
            mc = d["coords"]
            ch_names = [str(n) for n in d["ch_names"]] if "ch_names" in d else []

            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            ax.scatter(vc[hemi == 0, 0], vc[hemi == 0, 1], vc[hemi == 0, 2],
                       s=1, alpha=0.15, c="steelblue")
            ax.scatter(vc[hemi == 1, 0], vc[hemi == 1, 1], vc[hemi == 1, 2],
                       s=1, alpha=0.15, c="tomato")
            ax.scatter(mc[:, 0], mc[:, 1], mc[:, 2], s=20, c="gold", zorder=5)
            ax.set_title(mont_id, fontsize=8)
            ax.set_xlabel("x", fontsize=7)
            ax.set_ylabel("y", fontsize=7)
            ax.set_zlabel("z", fontsize=7)
            ax.tick_params(labelsize=6)

        plt.tight_layout()
        plt.show()

        # MNE alignment (fsaverage / mne_sample only)
        for mont_id in shown_montages:
            elec_npz  = lf_dir / anat_id / mont_id / "electrode_coords.npz"
            mont_path = mont_dir / f"{mont_id}.npz"
            src = elec_npz if elec_npz.exists() else mont_path if mont_path.exists() else None
            if src is None:
                continue
            d = np.load(src, allow_pickle=False)
            mc = d["coords"]
            ch_names = [str(n) for n in d["ch_names"]] if "ch_names" in d else [
                f"EEG{j:03d}" for j in range(len(mc))
            ]
            _try_mne_alignment(anat_id, mont_id, mc, ch_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EEG bank files.")
    parser.add_argument(
        "--banks-dir", type=Path, default=Path("banks"),
        help="Path to banks/ directory (default: banks/)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show diagnostic plots (requires matplotlib)",
    )
    args = parser.parse_args()

    banks_dir = args.banks_dir.resolve()
    if not banks_dir.exists():
        print(f"ERROR: banks directory not found: {banks_dir}")
        sys.exit(1)

    total_pass = total_warn = total_fail = 0

    def _print_section(title: str, results: list[dict]) -> None:
        nonlocal total_pass, total_warn, total_fail
        print(f"\n=== {title} ===")
        if not results:
            print("  (none found)")
            return
        for r in results:
            status = r["status"]
            detail = f"  [{r['detail']}]" if r.get("detail") else ""
            if title == "ANATOMY BANKS":
                line = (f"  {r['name']:<20} N={r.get('N','?'):<6} "
                        f"{r.get('coord_range',''):<44} "
                        f"hemi {r.get('hemi',''):<15} "
                        f"parcels={r.get('parcels','?'):<30} {status}{detail}")
            elif title == "MONTAGE BANKS":
                line = (f"  {r['name']:<25} C={r.get('C','?'):<5} "
                        f"{r.get('coord_range',''):<44} {status}{detail}")
            elif title == "LEADFIELD BANKS":
                line = (f"  {r['name']:<55} G={r.get('shape','?'):<14} "
                        f"col_norm {r.get('col_norm',''):<48} "
                        f"NaN={r.get('NaN','?')} {status}{detail}")
            else:
                line = (f"  {r['name']:<55} {r.get('c_match',''):<20} "
                        f"{r.get('n_match',''):<20} {status}{detail}")
            print(line)
            if status == "PASS":
                total_pass += 1
            elif status == "WARN":
                total_warn += 1
            else:
                total_fail += 1

    anat_results, _ = check_anatomies(banks_dir)
    mont_results, _ = check_montages(banks_dir)
    lf_results, _ = check_leadfields(banks_dir)
    coh_results, _ = check_coherence(banks_dir)

    _print_section("ANATOMY BANKS", anat_results)
    _print_section("MONTAGE BANKS", mont_results)
    _print_section("LEADFIELD BANKS", lf_results)
    _print_section("CROSS-COHERENCE", coh_results)

    print(f"\nSummary: {total_pass} checks PASS, {total_warn} WARN, {total_fail} FAIL")

    if args.plot:
        _plot_banks(banks_dir)

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
