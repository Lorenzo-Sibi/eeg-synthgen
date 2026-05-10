import argparse
import datetime as _dt
import importlib.metadata as _md
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import zarr

import mne
from mne.beamformer import apply_lcmv, make_lcmv
from mne.minimum_norm import apply_inverse, make_inverse_operator

from synthgen.analysis.inverse_metrics import SurfaceGeometry, compute_all_metrics
from synthgen.banks.leadfield import LeadfieldBank
from synthgen.config import GenerationConfig


# Zarr stores EEG in nV; MNE expects V.
EEG_NV_TO_V = 1e-9

DEFAULT_METHODS = ["MNE", "dSPM", "sLORETA", "eLORETA", "LCMV"]
DEFAULT_METRICS = ["le_mm", "te_ms", "nmse", "psnr_db", "auc"]


def resolve_output_dir(dataset_dir: Path, override: Path | None) -> Path:
    """Return the directory where validation artifacts will be written.

    Default layout: <dataset_dir>/eval/valid_<YYYY-MM-DD_HH-MM-SS>/.
    """
    if override is not None:
        out = override
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = dataset_dir / "eval" / f"valid_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_datasets(path: Path) -> list[Path]:
    if (path / "metadata.jsonl").exists() and (path / "data.zarr").exists():
        return [path]
    datasets = [
        child
        for child in sorted(path.iterdir())
        if child.is_dir() and (child / "metadata.jsonl").exists() and (child / "data.zarr").exists()
    ]
    if not datasets:
        raise FileNotFoundError(f"No dataset found under {path}")
    return datasets


def read_metadata(dataset_dir: Path) -> list[dict]:
    with (dataset_dir / "metadata.jsonl").open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_info(leadfield, sfreq: float) -> mne.Info:
    info = mne.create_info(leadfield.ch_names, sfreq=sfreq, ch_types="eeg")
    ch_pos = {
        name: xyz.astype(float) / 1000.0
        for name, xyz in zip(leadfield.ch_names, leadfield.electrode_coords)
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage, on_missing="ignore")

    dummy = mne.EvokedArray(np.zeros((len(info.ch_names), 1)), info, tmin=0.0, verbose=False)
    dummy.set_eeg_reference("average", projection=True, verbose=False)
    return dummy.info


def read_forward(meta: dict, forward_dir: Path, fixed: bool) -> mne.Forward:
    path = forward_dir / meta["anatomy_id"] / meta["montage_id"] / "forward-fwd.fif"
    if not path.exists():
        raise FileNotFoundError(f"Forward solution not found: {path}")

    fwd = mne.read_forward_solution(path, verbose=False)
    if fixed:
        return mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    return fwd


def surface_geometry(fwd: mne.Forward) -> SurfaceGeometry:
    coords, areas = [], []

    for src in fwd["src"]:
        vertno = np.asarray(src["vertno"], dtype=int)
        xyz = np.asarray(src["rr"][vertno], dtype=float) * 1000.0
        area = np.ones(len(vertno), dtype=float)

        tris = np.asarray(src.get("use_tris", []), dtype=int)
        if len(tris):
            area[:] = 0.0
            tri_xyz = xyz[tris]
            tri_area = 0.5 * np.linalg.norm(
                np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0]),
                axis=1,
            )
            np.add.at(area, tris.ravel(), np.repeat(tri_area / 3.0, 3))

        coords.append(xyz)
        areas.append(area)

    return SurfaceGeometry(
        vertex_coords_mm=np.concatenate(coords).astype(np.float32),
        vertex_areas_mm2=np.concatenate(areas).astype(np.float32),
    )


def make_mne_state(meta: dict, sfreq: float, leadfield_bank: LeadfieldBank, forward_dir: Path):
    """Build the shared MNE state for a (montage, anatomy) subset.

    Returns: (info, inverse_op, geometry, adjacency, fwd_fixed)
    All inverse methods (MNE/dSPM/sLORETA/eLORETA/LCMV) operate on the
    SAME fixed-orientation forward — see plan §7.4 for the rationale
    (Hauk 2022 convention; eliminates apples-vs-oranges between
    free-orientation LCMV and fixed-orientation MNE-family).
    """
    leadfield = leadfield_bank.load(meta["leadfield_id"])
    info = make_info(leadfield, sfreq)
    fwd_fixed = read_forward(meta, forward_dir, fixed=True)

    inverse_op = make_inverse_operator(
        info,
        fwd_fixed,
        mne.make_ad_hoc_cov(info),
        loose=0.0,
        fixed=True,
        depth=None,
        verbose=False,
    )
    geometry = surface_geometry(fwd_fixed)
    adjacency = mne.spatial_src_adjacency(fwd_fixed["src"], verbose=False).tocsr()

    return info, inverse_op, geometry, adjacency, fwd_fixed


def empirical_covariance(evoked: mne.EvokedArray, reg: float = 1e-6) -> mne.Covariance:
    data = evoked.data.astype(float)
    data = data - data.mean(axis=1, keepdims=True)
    cov = data @ data.T / max(data.shape[1] - 1, 1)
    cov += np.eye(cov.shape[0]) * max(np.trace(cov) / cov.shape[0], np.finfo(float).eps) * reg
    return mne.Covariance(cov, evoked.info.ch_names, evoked.info["bads"], evoked.info["projs"], data.shape[1])


_METHOD_CANONICAL = {
    "mne": "MNE",
    "dspm": "dSPM",
    "sloreta": "sLORETA",
    "eloreta": "eLORETA",
    "lcmv": "LCMV",
}


def apply_method(method: str, evoked: mne.EvokedArray, inverse_op,
                 fwd_fixed: mne.Forward, snr_db: float):
    """Dispatch to the appropriate MNE inverse routine.

    All five methods use the same fixed-orientation forward (see
    make_mne_state). LCMV uses pick_ori='max-power' (Hauk 2022): with a
    fixed-orientation forward this degenerates to a single-orientation
    beamformer, so the output is always a scalar SourceEstimate.
    """
    canonical = _METHOD_CANONICAL.get(method.lower(), method)

    if canonical == "LCMV":
        filters = make_lcmv(
            evoked.info,
            fwd_fixed,
            empirical_covariance(evoked),
            reg=0.05,
            noise_cov=mne.make_ad_hoc_cov(evoked.info),
            pick_ori="max-power",
            weight_norm="unit-noise-gain",
            reduce_rank=True,
            verbose=False,
        )
        return apply_lcmv(evoked, filters, verbose=False)

    snr_amp = 10.0 ** (float(snr_db) / 20.0)
    return apply_inverse(
        evoked, inverse_op,
        lambda2=1.0 / snr_amp ** 2,
        method=canonical,
        verbose=False,
    )


def evaluate_subset(
    store, group_key: str, metadata: list[dict], args, sfreq, leadfield_bank
) -> tuple[list[dict], dict]:
    """Evaluate every (sample, method) pair in a subset.

    Returns: (records, mne_state). `mne_state` exposes the MNE objects
    so the optional brain-plot helper (Task 16) can reuse them without
    rebuilding the inverse operator.
    """
    print(
        f"  {metadata[0]['montage_id']}: {len(metadata)} sample(s)"
        + (f" (n_samples={args.n_samples})" if args.n_samples is not None else "")
    )
    info, inverse_op, geometry, adjacency, fwd_fixed = make_mne_state(
        metadata[0], sfreq, leadfield_bank, args.forward_dir
    )

    group = store[group_key]
    sa       = group["source_activity"]  # (N, V, T)
    supports = group["source_support"]   # (N, V)
    eeg      = group["eeg"]              # (N, C, T)

    n_total = sa.shape[0]
    if args.n_samples is not None and args.n_samples < n_total:
        indices = np.random.choice(n_total, size=args.n_samples, replace=False)
        indices.sort()  # sequential is better on zarr chunked
        metadata = [metadata[i] for i in indices]
    else:
        indices = np.arange(n_total)

    records: list[dict] = []
    for sample_idx, (zarr_idx, meta) in enumerate(zip(indices, metadata)):
        zarr_idx = int(zarr_idx)

        evoked = mne.EvokedArray(
            np.asarray(eeg[zarr_idx]).astype(float) * EEG_NV_TO_V,
            info.copy(),
            tmin=0.0,
            verbose=False,
        )
        support = np.asarray(supports[zarr_idx], dtype=bool)
        j_true = np.asarray(sa[zarr_idx], dtype=float)
        seed_indices = np.asarray(meta["seed_vertex_indices"], dtype=int)

        for method in args.methods:
            stc = apply_method(method, evoked, inverse_op, fwd_fixed, meta["snr_sensor_db"])
            j_hat = np.asarray(stc.data, dtype=float)
            assert j_true.shape == j_hat.shape, (
                f"Shape mismatch GT vs {method}: {j_true.shape} vs {j_hat.shape}"
            )

            metrics = compute_all_metrics(
                j_true=j_true,
                j_hat=j_hat,
                seed_indices=seed_indices,
                support=support,
                coords_mm=geometry.vertex_coords_mm,
                adjacency=adjacency,
                sfreq=sfreq,
                patch_order=2,
            )

            records.append({
                "dataset":      str(args.current_dataset),
                "sample_idx":   sample_idx,
                "zarr_idx":     zarr_idx,
                "montage_id":   meta["montage_id"],
                "anatomy_id":   meta["anatomy_id"],
                "snr_db":       float(meta["snr_sensor_db"]),
                "snr_bin_db":   int(np.floor(float(meta["snr_sensor_db"]) / 5.0) * 5),
                "n_sources":    int(meta.get("n_sources", 1)),
                "prior_family": meta.get("prior_family", ""),
                "method":       method,
                **metrics,
            })

    mne_state = dict(
        info=info,
        inverse_op=inverse_op,
        geometry=geometry,
        adjacency=adjacency,
        fwd_fixed=fwd_fixed,
    )
    return records, mne_state


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_LIST_COLUMNS = {
    "per_seed_le_mm", "per_seed_te_ms", "per_seed_nmse", "per_seed_auc",
    "true_seed_indices", "pred_seed_indices",
    "t_eval_gt_indices", "t_eval_pred_indices",
}

_DIRECTION = {"le_mm": "↓", "te_ms": "↓", "nmse": "↓", "psnr_db": "↑", "auc": "↑"}
_METRIC_HUMAN = {
    "le_mm":   "LE [mm]",
    "te_ms":   "TE [ms]",
    "nmse":    "nMSE",
    "psnr_db": "PSNR [dB]",
    "auc":     "AUC",
}


def write_records_csv(records: list[dict], path: Path) -> None:
    """Long-format CSV. List-valued columns are JSON-encoded for safe round-trip."""
    df = pd.DataFrame.from_records(records)
    for col in _LIST_COLUMNS & set(df.columns):
        df[col] = df[col].apply(json.dumps)
    df.to_csv(path, index=False)


def write_summary_markdown(
    records: list[dict], path: Path, metrics: list[str] = DEFAULT_METRICS
) -> None:
    """Paper-style markdown table: mean ± std | median [Q25, Q75], one row per method."""
    df = pd.DataFrame.from_records(records)
    rows: list[dict] = []
    for method, sub in df.groupby("method"):
        row: dict = {"method": method}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            if len(vals) == 0:
                row[m] = "—"
                continue
            row[m] = (
                f"{vals.mean():.3g} ± {vals.std():.3g}"
                f" | {vals.median():.3g} [{vals.quantile(0.25):.3g}, {vals.quantile(0.75):.3g}]"
            )
        rows.append(row)
    summary = pd.DataFrame(rows)

    header = ["Method"] + [f"{_METRIC_HUMAN[m]} {_DIRECTION[m]}" for m in metrics]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for _, row in summary.iterrows():
        cells = [str(row["method"])] + [str(row[m]) for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text(
        "# Summary by method\n\n"
        "Format: mean ± std | median [Q25, Q75]\n\n"
        + "\n".join(lines)
        + "\n"
    )


def write_crosstab_csv(
    records: list[dict], path: Path, group_col: str,
    metrics: list[str] = DEFAULT_METRICS,
) -> None:
    """Cross-tab CSV with mean/std/median/q25/q75 per (group_col, method)."""
    df = pd.DataFrame.from_records(records)
    out_rows: list[dict] = []
    for (group_val, method), sub in df.groupby([group_col, "method"]):
        row: dict = {group_col: group_val, "method": method, "n": len(sub)}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            if len(vals):
                row[f"{m}_mean"]   = float(vals.mean())
                row[f"{m}_std"]    = float(vals.std())
                row[f"{m}_median"] = float(vals.median())
                row[f"{m}_q25"]    = float(vals.quantile(0.25))
                row[f"{m}_q75"]    = float(vals.quantile(0.75))
            else:
                for suffix in ("mean", "std", "median", "q25", "q75"):
                    row[f"{m}_{suffix}"] = float("nan")
        out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(path, index=False)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            timeout=2.0,
        ).strip()
    except Exception:
        return "unknown"


def dump_run_config(
    args: argparse.Namespace,
    config_obj,
    path: Path,
    n_records: int,
    started_at: str,
    finished_at: str,
) -> None:
    """Write a JSON snapshot of CLI args, config, git rev, package versions."""
    versions: dict[str, str] = {}
    for pkg in ("mne", "numpy", "scipy", "scikit-learn", "matplotlib", "pandas", "zarr"):
        try:
            versions[pkg] = _md.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    payload = {
        "cli_args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "config": config_obj.model_dump() if hasattr(config_obj, "model_dump") else None,
        "git_revision": _git_revision(),
        "package_versions": versions,
        "n_records": n_records,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Optional brain plots
# ---------------------------------------------------------------------------

def save_worst_le_brain_plots(
    records: list[dict],
    output_dir: Path,
    datasets_state: dict,
    eeg_data_by_dataset: dict,
) -> None:
    """For each (montage, method) save one PNG with the worst-LE sample.

    The brain is rendered at the time of the GT peak of the first seed
    (t_eval_gt[0]) for visual coherence with the LE metric.

    `datasets_state` maps dataset_dir → mne_state dict from evaluate_subset.
    `eeg_data_by_dataset` maps dataset_dir → (sa, supports, eeg) zarr handles.
    """
    df = pd.DataFrame.from_records(records)
    brain_dir = output_dir / "brain_plots"
    brain_dir.mkdir(exist_ok=True)

    for (montage, method), sub in df.groupby(["montage_id", "method"]):
        if sub["le_mm"].dropna().empty:
            continue
        worst = sub.loc[sub["le_mm"].idxmax()]
        dataset_dir = Path(worst["dataset"])
        state = datasets_state.get(dataset_dir)
        if state is None:
            continue
        _sa, _supports, eeg = eeg_data_by_dataset[dataset_dir]
        zarr_idx = int(worst["zarr_idx"])

        evoked = mne.EvokedArray(
            np.asarray(eeg[zarr_idx]).astype(float) * EEG_NV_TO_V,
            state["info"].copy(),
            tmin=0.0,
            verbose=False,
        )
        snr_db = float(worst["snr_db"])
        stc = apply_method(method, evoked, state["inverse_op"], state["fwd_fixed"], snr_db)

        true_seeds = json.loads(worst["true_seed_indices"])
        pred_seeds = json.loads(worst["pred_seed_indices"])
        t_gt_indices = json.loads(worst["t_eval_gt_indices"])

        n_lh = len(stc.vertices[0])
        initial_time = float(stc.times[t_gt_indices[0]])
        brain = stc.plot(
            hemi="both", clim="auto",
            initial_time=initial_time, time_unit="s",
            size=(800, 800), smoothing_steps=10,
        )
        for seed_row, pred_row in zip(true_seeds, pred_seeds):
            for row, color, alpha in [(seed_row, "lime", 0.9), (pred_row, "orange", 0.6)]:
                if row < n_lh:
                    hemi, mesh_v = "lh", int(stc.vertices[0][row])
                else:
                    hemi, mesh_v = "rh", int(stc.vertices[1][row - n_lh])
                brain.add_foci(
                    mesh_v, coords_as_verts=True, hemi=hemi,
                    color=color, scale_factor=0.6, alpha=alpha,
                )

        brain.add_text(
            0.05, 0.92,
            f"{method} worst LE = {worst['le_mm']:.1f} mm — {montage}",
            "title", font_size=12,
        )
        brain.save_image(str(brain_dir / f"worst_le_{montage}_{method}.png"))
        brain.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate classical EEG inverse methods on synthgen data."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/validate_classic_fsaverage.yaml"),
    )
    parser.add_argument("--forward-dir", type=Path, default=Path("banks/leadfield"))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--montages", nargs="+",
        help="Optional montage_id filter; only matching subsets are evaluated.",
    )
    parser.add_argument(
        "--n-samples", type=int,
        help="Per-montage sample cap. If omitted, all samples are used.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help=(
            "Where to write CSV/MD/PDF artifacts. "
            "Default: <dataset_dir>/eval/valid_<timestamp>/"
        ),
    )
    parser.add_argument(
        "--save-brain-plots", action="store_true",
        help="Save per-(montage,method) brain plot of worst-LE sample.",
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Skip ANOVA + Tukey-HSD computation.",
    )
    return parser.parse_args()


def main() -> None:
    from synthgen.analysis.plotting import boxplot_per_montage, boxplot_per_snr_bin
    from synthgen.analysis.stat_tests import anova_tukey_per_metric

    args = parse_args()
    config = GenerationConfig.from_yaml(str(args.config))
    leadfield_bank = LeadfieldBank(config.leadfield_bank)

    montage_filter = set(args.montages) if args.montages else None
    output_dir = resolve_output_dir(args.dataset_dir, args.output_dir)
    started_at = _dt.datetime.now().isoformat(timespec="seconds")

    records: list[dict] = []
    datasets_state: dict[Path, dict] = {}
    eeg_data_by_dataset: dict[Path, tuple] = {}
    for dataset_dir in find_datasets(args.dataset_dir):
        print(f"\nDataset: {dataset_dir}")
        args.current_dataset = dataset_dir

        store = zarr.open(str(dataset_dir / "data.zarr"), mode="r")
        metadata = read_metadata(dataset_dir)

        if montage_filter and metadata[0]["montage_id"] not in montage_filter:
            print(f"  skipped (not in --montages filter)")
            continue

        group_key = f"{metadata[0]['anatomy_id']}__{metadata[0]['montage_id']}"
        assert all(m["montage_id"] == metadata[0]["montage_id"] for m in metadata)
        assert all(m["anatomy_id"] == metadata[0]["anatomy_id"] for m in metadata)
        assert all(m["leadfield_id"] == metadata[0]["leadfield_id"] for m in metadata)

        subset_records, mne_state = evaluate_subset(
            store, group_key, metadata, args, config.temporal.sfreq, leadfield_bank,
        )
        records.extend(subset_records)

        if args.save_brain_plots:
            datasets_state[dataset_dir] = mne_state
            zgroup = store[group_key]
            eeg_data_by_dataset[dataset_dir] = (
                zgroup["source_activity"],
                zgroup["source_support"],
                zgroup["eeg"],
            )

    if not records:
        raise RuntimeError("No records evaluated. Check --dataset-dir and --montages.")

    # Long-format records and summary tables
    write_records_csv(records, output_dir / "records.csv")
    write_summary_markdown(records, output_dir / "summary_by_method.md")
    write_crosstab_csv(records, output_dir / "summary_by_montage_method.csv", "montage_id")
    write_crosstab_csv(records, output_dir / "summary_by_snr_method.csv", "snr_bin_db")

    # ANOVA + Tukey-HSD pairwise per metric
    if not args.no_stats:
        tukey_df = anova_tukey_per_metric(records, metrics=DEFAULT_METRICS)
        tukey_df.to_csv(output_dir / "anova_tukey.csv", index=False)

    # Publication-quality boxplots
    boxplot_per_montage(records, DEFAULT_METRICS, output_dir)
    boxplot_per_snr_bin(records, DEFAULT_METRICS, output_dir)

    # Optional brain plots of the worst-LE sample per (montage, method)
    if args.save_brain_plots:
        save_worst_le_brain_plots(
            records, output_dir, datasets_state, eeg_data_by_dataset,
        )

    # Run snapshot for reproducibility
    finished_at = _dt.datetime.now().isoformat(timespec="seconds")
    dump_run_config(
        args, config, output_dir / "run_config.json",
        len(records), started_at, finished_at,
    )

    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()