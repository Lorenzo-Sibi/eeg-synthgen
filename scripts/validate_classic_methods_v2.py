import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import zarr
from scipy.spatial.distance import cdist

import mne
from mne.beamformer import apply_lcmv, make_lcmv
from mne.minimum_norm import apply_inverse, make_inverse_operator

from synthgen.banks.leadfield import LeadfieldBank
from synthgen.config import GenerationConfig


"""
"auc",
"nmse",
"recall",
"precision",
"f1",
"spatial_dispersion_mm",
"""
METRICS = [
    "le_mm",
]

"""
"auc": "AUC",
"nmse": "nMSE",
"recall": "Recall",
"precision": "Prec",
"f1": "F1",
"spatial_dispersion_mm": "SD(mm)",
"""
METRIC_LABELS = {
    "le_mm": "LE(mm)",
}

EEG_NV_TO_V = 1e-9


@dataclass(frozen=True)
class SurfaceGeometry:
    vertex_coords_mm: np.ndarray
    vertex_areas_mm2: np.ndarray


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


def montage_sort_key(name: str) -> tuple[int, str]:
    montage = name.split("__")[-1]
    suffix = montage.rsplit("_", 1)[-1]
    return (int(suffix), montage) if suffix.isdigit() else (10**9, montage)


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


def get_patch(seed_idx: int, adjacency, order: int = 2) -> np.ndarray:
    """
    Build an order-k local patch around a seed using an MNE sparse adjacency matrix.

    Parameters
    ----------
    seed_idx:
        Source-space row index of the true seed.

    adjacency:
        Sparse adjacency matrix returned by mne.spatial_src_adjacency(fwd["src"]).

    order:
        Neighborhood order. stESI uses order=2.

    Returns
    -------
    patch:
        Array of source indices in the local patch.
    """
    patch = {int(seed_idx)}
    frontier = {int(seed_idx)}

    for _ in range(order):
        new_frontier = set()

        for idx in frontier:
            neighbors = adjacency.indices[adjacency.indptr[idx]:adjacency.indptr[idx + 1]]
            new_frontier.update(int(n) for n in neighbors)

        patch.update(new_frontier)
        frontier = new_frontier

    return np.array(sorted(patch), dtype=int)


def compute_le(
    j_true: np.ndarray,
    j_hat: np.ndarray,
    seed_vertex_indices: list[int] | np.ndarray,
    coords_mm: np.ndarray,
    adjacency,
    patch_order: int = 2,
) -> dict:
    """
    Compute Localization Error (LE) as in the https://github.com/SarahReynaud/stESI_pub evaluation pipeline.
    stESI_pub is the code of the paper: "Comprehensive analysis of supervised learning methods for electrical source imaging" by S. Reynaud et al.
    For each true seed s:
        1. t_eval_gt = argmax_t |j_true[s, t]|
        2. eval_zone = local order-2 patch around s
        3. s_hat = argmax_v |j_hat[v, t_eval_gt]| inside eval_zone
        4. LE_s = Euclidean distance between s and s_hat

    The final LE is the mean over all true seeds.
    """
    j_true = np.asarray(j_true, dtype=float)
    j_hat = np.asarray(j_hat, dtype=float)
    coords_mm = np.asarray(coords_mm, dtype=float)
    seed_vertex_indices = np.asarray(seed_vertex_indices, dtype=int)

    if j_true.ndim != 2:
        raise ValueError(f"j_true must have shape (V, T), got {j_true.shape}")
    if j_hat.ndim != 2:
        raise ValueError(f"j_hat must have shape (V, T), got {j_hat.shape}")
    if j_true.shape != j_hat.shape:
        raise ValueError(f"j_true and j_hat must have same shape, got {j_true.shape} and {j_hat.shape}")
    if coords_mm.shape[0] != j_true.shape[0]:
        raise ValueError(f"coords_mm has {coords_mm.shape[0]} vertices, but j_true has {j_true.shape[0]}")

    le_values = []
    pred_vertex_indices = []
    t_eval_gt_indices = []

    for seed_idx in seed_vertex_indices:
        seed_idx = int(seed_idx)

        # 1. Time of maximum true source activity for this seed
        t_eval_gt = int(np.argmax(np.abs(j_true[seed_idx, :])))

        # 2. Local evaluation patch around the true seed
        eval_zone = get_patch(
            seed_idx=seed_idx,
            adjacency=adjacency,
            order=patch_order,
        )

        # 3. Strongest estimated source inside local patch at t_eval_gt
        local_values = np.abs(j_hat[eval_zone, t_eval_gt])
        pred_idx = int(eval_zone[np.argmax(local_values)])

        # 4. Euclidean localization error
        le_mm = float(np.linalg.norm(coords_mm[seed_idx] - coords_mm[pred_idx]))

        le_values.append(le_mm)
        pred_vertex_indices.append(pred_idx)
        t_eval_gt_indices.append(t_eval_gt)

    return {
        "le_mm": float(np.mean(le_values)),
        "per_seed_le_mm": le_values,
        "true_seed_vertex_indices": seed_vertex_indices.tolist(),
        "pred_seed_vertex_indices": pred_vertex_indices,
        "t_eval_gt_indices": t_eval_gt_indices,
    }
    

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


def make_mne_state(meta: dict, sfreq: float, leadfield_bank: LeadfieldBank, forward_dir: Path, methods: list[str]):
    leadfield = leadfield_bank.load(meta["leadfield_id"])
    info = make_info(leadfield, sfreq)
    fixed_fwd = read_forward(meta, forward_dir, fixed=True)

    inverse = make_inverse_operator(
        info,
        fixed_fwd,
        mne.make_ad_hoc_cov(info),
        loose=0.0,
        fixed=True,
        depth=None,
        verbose=False,
    )
    lcmv_fwd = read_forward(meta, forward_dir, fixed=False) if "LCMV" in {m.upper() for m in methods} else None
    geometry = surface_geometry(fixed_fwd)
    adjacency = mne.spatial_src_adjacency(fixed_fwd["src"], verbose=False).tocsr()
    
    return info, inverse, lcmv_fwd, geometry, adjacency


def empirical_covariance(evoked: mne.EvokedArray, reg: float = 1e-6) -> mne.Covariance:
    data = evoked.data.astype(float)
    data = data - data.mean(axis=1, keepdims=True)
    cov = data @ data.T / max(data.shape[1] - 1, 1)
    cov += np.eye(cov.shape[0]) * max(np.trace(cov) / cov.shape[0], np.finfo(float).eps) * reg
    return mne.Covariance(cov, evoked.info.ch_names, evoked.info["bads"], evoked.info["projs"], data.shape[1])


def apply_method(method: str, evoked: mne.EvokedArray, inverse, lcmv_fwd, snr_db: float):
    if method.upper() == "LCMV":
        filters = make_lcmv(
            evoked.info,
            lcmv_fwd,
            empirical_covariance(evoked),
            reg=0.05,
            noise_cov=mne.make_ad_hoc_cov(evoked.info),
            pick_ori=None,
            weight_norm="unit-noise-gain",
            reduce_rank=True,
            verbose=False,
        )
        return apply_lcmv(evoked, filters, verbose=False)

    snr = 10.0 ** (float(snr_db) / 20.0)
    return apply_inverse(evoked, inverse, lambda2=1.0 / snr**2, method=method, verbose=False)


def source_scores(stc) -> np.ndarray:
    data = np.abs(np.asarray(stc.data))
    if data.ndim == 1:
        return data
    return data.max(axis=tuple(range(1, data.ndim)))


def otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.inf
    if np.allclose(values.min(), values.max()):
        return float(values.max())

    hist, edges = np.histogram(values, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(hist.sum())
    w0 = np.cumsum(hist) / total
    w1 = 1.0 - w0
    mu0 = np.cumsum(hist * centers) / (np.cumsum(hist) + 1e-20)
    # mu1 = np.cumsum(hist * centers) / ...
    mu_total = float(np.sum(hist * centers)) / total
    mu1 = (mu_total - w0 * mu0) / (w1 + 1e-20)
    return float(centers[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(np.sum(values * weights) / total) if total > 0.0 else np.nan


def compute_paper_metrics(scores: np.ndarray, support: np.ndarray, geometry: SurfaceGeometry) -> dict:
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    support = np.asarray(support, dtype=bool)
    coords = geometry.vertex_coords_mm
    areas = geometry.vertex_areas_mm2.astype(float)

    if scores.shape != support.shape or scores.shape != areas.shape:
        raise ValueError(f"Shape mismatch: scores={scores.shape}, support={support.shape}, areas={areas.shape}")

    estimated = scores > otsu_threshold(scores)
    overlap = support & estimated

    gt_area = float(areas[support].sum())
    est_area = float(areas[estimated].sum())
    overlap_area = float(areas[overlap].sum())

    recall = overlap_area / gt_area if gt_area else np.nan
    precision = overlap_area / est_area if est_area else np.nan
    f1 = 2.0 * recall * precision / (recall + precision + 1e-12)

    if not support.any() or not estimated.any():
        le_mm = np.nan
        sd_mm = np.nan
    else:
        distances = cdist(coords[support], coords[estimated])
        gt_to_est = distances.min(axis=1)
        est_to_gt = distances.min(axis=0)
        le_mm = 0.5 * (weighted_mean(gt_to_est, areas[support]) + weighted_mean(est_to_gt, areas[estimated]))

        sd_weights = areas[estimated] * scores[estimated] ** 2
        sd_mm = float(np.sqrt(np.sum(est_to_gt**2 * sd_weights) / (sd_weights.sum() + 1e-20)))

    return {
        "le_mm": le_mm,
        "recall": recall,
        "precision": precision,
        "harmonic_mean": f1,
        "sd_mm": sd_mm,
        "gt_area_cm2": gt_area / 100.0,
        "estimated_area_cm2": est_area / 100.0,
        "overlap_area_cm2": overlap_area / 100.0,
    }


def evaluate_subset(store, group_key: str, metadata: list[dict], args, sfreq, leadfield_bank) -> list[dict]:
    print(f"  {metadata[0]['montage_id']}: {len(metadata)} total sample(s)", f"of which {args.n_samples} selected" if args.n_samples is not None else "")
    info, inverse, lcmv_fwd, geometry, adjacency = make_mne_state(metadata[0], sfreq, leadfield_bank, args.forward_dir, args.methods)

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
            
    worst_stc = dict()
    records = []
    for sample_idx, (zarr_idx, meta) in enumerate(zip(indices, metadata)):
        zarr_idx = int(zarr_idx) # access to a single one to avoid crash (zarr all load on memory when indexing)
        
        evoked = mne.EvokedArray(
            np.asarray(eeg[zarr_idx]).astype(float) * EEG_NV_TO_V, # (C, T)
            info.copy(),
            tmin=0.0,
            verbose=False,
        )

        support = np.asarray(supports[zarr_idx], dtype=bool)
        for method in args.methods:
            stc = apply_method(method, evoked, inverse, lcmv_fwd, meta["snr_sensor_db"])
            seed_vertex_indices = np.asarray(meta["seed_vertex_indices"])
            
            j_true = np.asarray(sa[zarr_idx], dtype=float)        # ground truth full source activity for the sample: (V, T)
            j_hat = np.asarray(stc.data, dtype=float)             # Estimated source activity from MNE/STC: (V, T)
            
            assert j_true.shape[0] == j_hat.shape[0] == geometry.vertex_coords_mm.shape[0] == adjacency.shape[0] # just in case...
            
            le_metrics = compute_le(
                j_true=j_true,
                j_hat=j_hat,
                seed_vertex_indices=seed_vertex_indices,
                coords_mm=geometry.vertex_coords_mm,
                adjacency=adjacency,
                patch_order=2,
            )
            
            if method not in worst_stc or le_metrics["le_mm"] > worst_stc[method][1]["le_mm"]:
                worst_stc[method] = tuple([stc, le_metrics])
            
            records.append({
                "dataset":      str(args.current_dataset),
                "sample_index": sample_idx,
                "montage_id":   meta["montage_id"],
                "snr_db":       float(meta["snr_sensor_db"]),
                "snr_bin_db":   int(np.floor(float(meta["snr_sensor_db"]) / 5.0) * 5),
                "n_sources":    int(meta.get("n_sources", 1)),
                "prior_family": meta.get("prior_family", ""),
                "method":       method,
                "le_mm": le_metrics["le_mm"],
                "per_seed_le_mm": le_metrics["per_seed_le_mm"],
                "true_seed_vertex_indices": le_metrics["true_seed_vertex_indices"],
                "pred_seed_vertex_indices": le_metrics["pred_seed_vertex_indices"],
                "t_eval_gt_indices": le_metrics["t_eval_gt_indices"],
                #**compute_paper_metrics(peak_values, support, geometry),
            })
            
        if args.show_plot and worst_stc:
            open_brains = []
            for method in worst_stc.keys():
                stc, le_metrics = worst_stc[method]
                vertno_max, time_max = stc.get_peak()
                brain = stc.plot(
                    hemi="both",
                    clim='auto',
                    initial_time=time_max,
                    time_unit="s",
                    size=(800, 800),
                    smoothing_steps=10,
                )
                open_brains.append(brain)

                # stc.vertices[0]/[1] hold *mesh* vertex numbers, while compute_le returns
                # *source-space row indices* (range [0, V)). Convert before passing to add_foci.
                n_lh = len(stc.vertices[0])
                for i, (seed_row, pred_row) in enumerate(zip(
                    le_metrics["true_seed_vertex_indices"],
                    le_metrics["pred_seed_vertex_indices"],
                )):
                    if pred_row < n_lh:
                        pred_hemi = "lh"
                        pred_mesh_vertex = int(stc.vertices[0][pred_row])
                    else:
                        pred_hemi = "rh"
                        pred_mesh_vertex = int(stc.vertices[1][pred_row - n_lh])

                    if seed_row < n_lh:
                        seed_hemi = "lh"
                        seed_mesh_vertex = int(stc.vertices[0][seed_row])
                    else:
                        seed_hemi = "rh"
                        seed_mesh_vertex = int(stc.vertices[1][seed_row - n_lh])

                    brain.add_foci(
                        seed_mesh_vertex,
                        coords_as_verts=True,
                        hemi=seed_hemi,
                        color="lime",
                        scale_factor=0.6,
                        alpha=0.8,
                    )
                    brain.add_foci(
                        pred_mesh_vertex,
                        coords_as_verts=True,
                        hemi=pred_hemi,
                        color="orange",
                        scale_factor=0.6,
                        alpha=0.5,
                    )
                    print(f"{method} - seed vertex index {seed_row}, predicted vertex index {pred_row}, LE for this seed: {le_metrics['per_seed_le_mm'][i]}")
                brain.add_text(0.1, 0.9, f"{method} worst Localization Error (LE) = {le_metrics['le_mm']} on montage {meta['montage_id']}", "title", font_size=14)

            # Block here so WSL doesn't accumulate brain windows across samples.
            try:
                input(
                    f"  Sample {sample_idx} ({meta['montage_id']}, "
                    f"snr={float(meta['snr_sensor_db']):.1f} dB) - "
                    f"Enter to close & continue, Ctrl+C to stop..."
                )
            except (KeyboardInterrupt, EOFError):
                for b in open_brains:
                    b.close()

    return records


def finite_stats(records: list[dict], field: str) -> tuple[float, float, float]:
    values = np.array([row[field] for row in records], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    return float(values.mean()), float(values.std()), float(np.median(values))


def print_summary(records: list[dict], keys: list[str], title: str) -> None:
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[key] for key in keys)].append(row)

    col_w = 26
    header = f"{'Group':<42} {'N':>5}" + "".join(
        f"  {METRIC_LABELS[m] + ' mean±std':>{col_w}}  {'median':>8}" for m in METRICS
    )
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for key, rows in sorted(groups.items()):
        name = " | ".join(map(str, key))
        stats = ""
        for m in METRICS:
            mean, std, median = finite_stats(rows, m)
            stats += f"  {f'{mean:.2f} ± {std:.2f}':>{col_w}}  {median:>8.2f}"
        print(f"{name:<42} {len(rows):>5}{stats}")


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV saved: {path}")


def show_plot(records: list[dict]):
    
    fig, ax = plt.figure()
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate classical EEG inverse methods on synthgen data.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/validate_classic_fsaverage.yaml"))
    parser.add_argument("--forward-dir", type=Path, default=Path("banks/leadfield"))
    parser.add_argument("--methods", nargs="+", default=["sLORETA", "LCMV"])
    parser.add_argument("--montages", nargs="+", help="Optional montage ids or zarr group keys.")
    parser.add_argument("--n-samples", type=int, help="Number of samples to randomly select per montage. If not provided, use all.")
    parser.add_argument("--show-plot", action='store_true')
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig.from_yaml(str(args.config))
    leadfield_bank = LeadfieldBank(config.leadfield_bank)

    records = []
    for dataset_dir in find_datasets(args.dataset_dir):
        print(f"\nDataset: {dataset_dir}")
        args.current_dataset = dataset_dir

        store = zarr.open(str(dataset_dir / "data.zarr"), mode="r")
        metadata = read_metadata(dataset_dir)
        group_key = f"{metadata[0]['anatomy_id']}__{metadata[0]['montage_id']}"
        
        assert all([m["montage_id"] == metadata[0]["montage_id"] for m in metadata]) # we want the subset to be packed with the same montage id
        assert all([m["anatomy_id"] == metadata[0]["anatomy_id"] for m in metadata]) # and anatomy id

        records.extend(evaluate_subset(store, group_key, metadata, args, config.temporal.sfreq, leadfield_bank))
        
    if not records:
        raise RuntimeError("No records evaluated. Check --dataset-dir and --montages.")

    print_summary(records, ["montage_id", "snr_bin_db", "method"], "By montage, SNR bin, method")
    print_summary(records, ["method"], "By method")
    
    if args.show_plot:
        show_plot(records)
    
    if args.output_csv:
        write_csv(records, args.output_csv)


if __name__ == "__main__":
    main()