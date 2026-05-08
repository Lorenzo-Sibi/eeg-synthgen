import argparse
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


import numpy as np
from numcodecs import Zstd
from scipy.spatial.distance import cdist

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import mne
from mne.beamformer import apply_lcmv, make_lcmv
from mne.minimum_norm import apply_inverse, make_inverse_operator

from synthgen.banks.leadfield import LeadfieldBank
from synthgen.config import GenerationConfig


METRICS = [
    ("le_mm", "LE(mm)", "Localization error (mm)"),
    ("recall", "Recall", "Recall"),
    ("precision", "Prec", "Precision"),
    ("harmonic_mean", "F1", "Harmonic mean"),
    ("sd_mm", "SD(mm)", "Spatial dispersion (mm)"),
]


@dataclass(frozen=True)
class SurfaceGeometry:
    vertex_coords_mm: np.ndarray
    vertex_areas_mm2: np.ndarray


@dataclass(frozen=True)
class EvalContext:
    info: mne.Info
    inverse_operator: object
    lcmv_forward: mne.Forward
    geometry: SurfaceGeometry


class MontageDataset:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.store_dir = self.dataset_dir / "data.zarr"
        self.metadata = self._load_metadata()
        if not self.metadata:
            raise ValueError(f"Dataset vuoto: {dataset_dir}")
        meta = self.metadata[0]
        self.group_key = f"{meta['anatomy_id']}__{meta['montage_id']}"
        self.montage_id = meta["montage_id"]

    def _load_metadata(self) -> list[dict]:
        with (self.dataset_dir / "metadata.jsonl").open("r") as f:
            return [json.loads(line) for line in f]

    def selected_indices(self, n_samples: int) -> range:
        return range(min(n_samples, len(self)))

    def read(self, index: int) -> dict:
        group = self.store_dir / self.group_key
        return {
            "eeg": read_zarr_v3_sample(group / "eeg", index),
            "source_support": read_zarr_v3_sample(group / "source_support", index),
            "metadata": self.metadata[index],
        }

    def __len__(self) -> int:
        return len(self.metadata)

def read_zarr_v3_sample(array_dir: Path, sample_index: int) -> np.ndarray:
    with (array_dir / "zarr.json").open("r") as f:
        meta = json.load(f)

    shape = tuple(map(int, meta["shape"]))
    chunk_shape = tuple(map(int, meta["chunk_grid"]["configuration"]["chunk_shape"]))
    if any(dim > chunk for dim, chunk in zip(shape[1:], chunk_shape[1:])):
        raise NotImplementedError(f"{array_dir} has non-sample chunking; this reader expects full trailing axes.")

    chunk_coords = [0] * len(shape)
    chunk_coords[0] = sample_index // chunk_shape[0]
    local_index = sample_index % chunk_shape[0]
    payload = (array_dir / "c" / Path(*map(str, chunk_coords))).read_bytes()

    for codec in reversed(meta.get("codecs", [])):
        if codec["name"] == "zstd":
            payload = Zstd(**codec.get("configuration", {})).decode(payload)
        elif codec["name"] != "bytes":
            raise NotImplementedError(f"Unsupported Zarr codec {codec['name']!r}")

    dtype = np.dtype(bool if meta["data_type"] == "bool" else meta["data_type"])
    chunk = np.frombuffer(payload, dtype=dtype).reshape(chunk_shape)
    sample = chunk[(local_index, *[slice(0, dim) for dim in shape[1:]])]
    return np.array(sample)


def make_info(leadfield, sfreq: float) -> mne.Info:
    info = mne.create_info(leadfield.ch_names, sfreq=sfreq, ch_types="eeg")
    ch_pos = {
        name: coord.astype(float) / 1000.0
        for name, coord in zip(leadfield.ch_names, leadfield.electrode_coords)
    }
    info.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head"), on_missing="ignore")

    dummy = mne.EvokedArray(np.zeros((len(info.ch_names), 1)), info, tmin=0.0, verbose=False)
    dummy.set_eeg_reference("average", projection=True, verbose=False)
    return dummy.info


def make_evoked(eeg: np.ndarray, info: mne.Info) -> mne.EvokedArray:
    return mne.EvokedArray(eeg.astype(float) * 1e-9, info.copy(), tmin=0.0, comment="synthgen", verbose=False)


def load_forward(forward_dir: Path, anatomy_id: str, montage_id: str, *, fixed: bool) -> mne.Forward:
    path = forward_dir / anatomy_id / montage_id / "forward-fwd.fif"
    if not path.exists():
        raise FileNotFoundError(f"Forward FIF not found: {path}")
    fwd = mne.read_forward_solution(path, verbose=False)
    return mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False) if fixed else fwd


def surface_geometry_from_forward(fwd: mne.Forward) -> SurfaceGeometry:
    coords, areas = [], []
    for src in fwd["src"]:
        vertno = np.asarray(src["vertno"], dtype=int)
        hemi_coords = np.asarray(src["rr"][vertno], dtype=float) * 1000.0
        hemi_areas = np.zeros(len(vertno), dtype=float)

        tris = np.asarray(src.get("use_tris", []), dtype=int)
        if len(tris):
            tri_coords = hemi_coords[tris]
            tri_areas = 0.5 * np.linalg.norm(
                np.cross(tri_coords[:, 1] - tri_coords[:, 0], tri_coords[:, 2] - tri_coords[:, 0]),
                axis=1,
            )
            np.add.at(hemi_areas, tris.ravel(), np.repeat(tri_areas / 3.0, 3))
        else:
            hemi_areas.fill(1.0)

        coords.append(hemi_coords)
        areas.append(hemi_areas)

    return SurfaceGeometry(
        np.concatenate(coords).astype(np.float32),
        np.concatenate(areas).astype(np.float32),
    )


def build_context(
    meta: dict,
    config: GenerationConfig,
    leadfield_bank: LeadfieldBank,
    forward_dir: Path,
    methods: list[str],
) -> EvalContext:
    leadfield = leadfield_bank.load(meta["leadfield_id"])
    info = make_info(leadfield, config.temporal.sfreq)
    fixed_fwd = load_forward(forward_dir, meta["anatomy_id"], meta["montage_id"], fixed=True)
    lcmv_fwd = (
        load_forward(forward_dir, meta["anatomy_id"], meta["montage_id"], fixed=False)
        if any(m.upper() == "LCMV" for m in methods)
        else fixed_fwd
    )
    inv = make_inverse_operator(
        info,
        fixed_fwd,
        mne.make_ad_hoc_cov(info),
        loose=0.0,
        fixed=True,
        depth=None,
        verbose=False,
    )
    return EvalContext(info, inv, lcmv_fwd, surface_geometry_from_forward(fixed_fwd))


def empirical_covariance(evoked: mne.EvokedArray, reg: float = 1e-6) -> mne.Covariance:
    data = evoked.data.astype(float)
    demeaned = data - data.mean(axis=1, keepdims=True)
    cov = demeaned @ demeaned.T / max(demeaned.shape[1] - 1, 1)
    cov += np.eye(cov.shape[0]) * max(float(np.trace(cov)) / cov.shape[0], np.finfo(float).eps) * reg
    return mne.Covariance(cov, evoked.info.ch_names, evoked.info["bads"], evoked.info["projs"], data.shape[1])


def apply_methods(evoked: mne.EvokedArray, ctx: EvalContext, methods: list[str], snr_db: float) -> dict:
    out = {}
    lambda2 = 1.0 / (10.0 ** (snr_db / 20.0)) ** 2
    for method in methods:
        if method.upper() == "LCMV":
            filters = make_lcmv(
                evoked.info,
                ctx.lcmv_forward,
                empirical_covariance(evoked),
                reg=0.05,
                noise_cov=mne.make_ad_hoc_cov(evoked.info),
                pick_ori=None,
                weight_norm="unit-noise-gain",
                reduce_rank=True,
                verbose=False,
            )
            out[method] = apply_lcmv(evoked, filters, verbose=False)
        else:
            out[method] = apply_inverse(evoked, ctx.inverse_operator, lambda2, method=method, verbose=False)
    return out


def stc_scores(stc) -> np.ndarray:
    return np.max(np.abs(stc.data), axis=1)


def otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.inf
    if np.allclose(values.min(), values.max()):
        return float(values.max())

    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(hist.sum())
    w0 = np.cumsum(hist) / total
    w1 = 1.0 - w0
    mu0 = np.cumsum(centers * hist) / (np.cumsum(hist) + 1e-20)
    mu_total = float((centers * hist).sum()) / total
    mu1 = (mu_total - w0 * mu0) / (w1 + 1e-20)
    return float(centers[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(weights.sum())
    return float(np.sum(values * weights) / denom) if denom > 0 else np.nan


def compute_paper_metrics(scores: np.ndarray, support: np.ndarray, geometry: SurfaceGeometry) -> dict:
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    support = np.asarray(support, dtype=bool)
    coords = geometry.vertex_coords_mm
    areas = geometry.vertex_areas_mm2.astype(float)
    if scores.shape != support.shape or scores.shape != areas.shape:
        raise ValueError(f"shape mismatch: scores={scores.shape}, support={support.shape}, areas={areas.shape}")

    estimated = scores > otsu_threshold(scores)
    overlap = support & estimated
    gt_area = float(areas[support].sum())
    est_area = float(areas[estimated].sum())
    overlap_area = float(areas[overlap].sum())

    recall = overlap_area / gt_area if gt_area else np.nan
    precision = overlap_area / est_area if est_area else np.nan
    f1 = 2 * recall * precision / (recall + precision + 1e-12)

    if not support.any() or not estimated.any():
        le = sd = np.nan
    else:
        distances = cdist(coords[support], coords[estimated])
        gt_to_est = distances.min(axis=1)
        est_to_gt = distances.min(axis=0)
        le = 0.5 * (
            weighted_mean(gt_to_est, areas[support])
            + weighted_mean(est_to_gt, areas[estimated])
        )
        sd_weights = areas[estimated] * scores[estimated] ** 2
        sd = float(np.sqrt(np.sum(est_to_gt**2 * sd_weights) / (sd_weights.sum() + 1e-20)))

    return {
        "le_mm": le,
        "recall": recall,
        "precision": precision,
        "harmonic_mean": f1,
        "sd_mm": sd,
        "gt_area_cm2": gt_area / 100.0,
        "estimated_area_cm2": est_area / 100.0,
        "overlap_area_cm2": overlap_area / 100.0,
    }


def evaluate_sample(sample: dict, ctx: EvalContext, methods: list[str]) -> list[dict]:
    meta = sample["metadata"]
    evoked = make_evoked(sample["eeg"], ctx.info)
    stcs = apply_methods(evoked, ctx, methods, meta["snr_sensor_db"])
    rows = []
    for method, stc in stcs.items():
        rows.append({
            "montage_id": meta["montage_id"],
            "snr_db": meta["snr_sensor_db"],
            "snr_bin_db": int(np.floor(meta["snr_sensor_db"] / 5.0) * 5),
            "n_sources": meta.get("n_sources", 1),
            "prior_family": meta.get("prior_family", "?"),
            "method": method,
            **compute_paper_metrics(stc_scores(stc), sample["source_support"], ctx.geometry),
        })
    return rows


def evaluate_dataset(
    dataset: MontageDataset,
    n_samples: int,
    config: GenerationConfig,
    leadfield_bank: LeadfieldBank,
    forward_dir: Path,
    methods: list[str],
) -> list[dict]:
    indices = dataset.selected_indices(n_samples)
    print(f"  {dataset.montage_id}: {len(indices)}/{len(dataset)} samples")
    ctx = build_context(dataset.metadata[0], config, leadfield_bank, forward_dir, methods)
    records = []
    for index in indices:
        records.extend(evaluate_sample(dataset.read(index), ctx, methods))
    return records


def finite_mean(rows: list[dict], field: str) -> float:
    values = np.array([r[field] for r in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def print_summary(records: list[dict]) -> None:
    def grouped(keys):
        groups = defaultdict(list)
        for record in records:
            groups[tuple(record[k] for k in keys)].append(record)
        return groups

    header = f"{'Montage':<22} {'SNR(dB)':>7} {'Method':<8} {'N':>4} " + " ".join(
        f"{short:>8}" for _, short, _ in METRICS
    )
    print(f"\n{'=== Metrics by montage x 5 dB SNR x method ':=<{len(header)}}")
    print(header)
    print("-" * len(header))
    for (montage, snr, method), rows in sorted(grouped(["montage_id", "snr_bin_db", "method"]).items()):
        vals = " ".join(f"{finite_mean(rows, field):>8.3g}" for field, _, _ in METRICS)
        print(f"{montage:<22} {snr:>7} {method:<8} {len(rows):>4} {vals}")

    print(f"\n{'=== Marginal by method ':=<{len(header)}}")
    for (method,), rows in sorted(grouped(["method"]).items()):
        vals = " ".join(f"{finite_mean(rows, field):>8.3g}" for field, _, _ in METRICS)
        print(f"{method:<8} N={len(rows):<4} {vals}")


def montage_sort_key(montage: str) -> tuple[int, str]:
    suffix = montage.rsplit("_", 1)[-1]
    return (int(suffix), montage) if suffix.isdigit() else (10**9, montage)


def short_label(value) -> str:
    if isinstance(value, tuple):
        return f"{short_label(value[0])}\n{value[1]} dB"
    suffix = str(value).rsplit("_", 1)[-1]
    return f"{suffix} ch" if suffix.isdigit() else str(value)


def plot_metric_grid(records: list[dict], group_fn, groups: list, title: str, path: Path | None, show: bool):
    if not records:
        return
    methods = sorted({r["method"] for r in records})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    centers = np.arange(len(groups))
    width = min(0.24, 0.75 / max(len(methods), 1))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for ax, (field, _, label) in zip(axes.ravel(), METRICS):
        for j, method in enumerate(methods):
            data = [metric_values(records, field, method, group_fn, group) for group in groups]
            pos = centers + offsets[j]
            box = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=False)
            for patch in box["boxes"]:
                patch.set(facecolor=colors[j % len(colors)], alpha=0.30, edgecolor=colors[j % len(colors)])
            ax.plot(pos, [finite_array_mean(v) for v in data], "o-", color=colors[j % len(colors)], label=method, ms=4)
        ax.set_title(label)
        ax.set_xticks(centers)
        ax.set_xticklabels([short_label(g) for g in groups], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if field in {"recall", "precision", "harmonic_mean"}:
            ax.set_ylim(-0.02, 1.02)

    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(frameon=False, title="Method")
    fig.suptitle(title)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        print(f"  Plot salvato: {path}")
    plt.show() if show else plt.close(fig)


def metric_values(records: list[dict], field: str, method: str, group_fn, group) -> np.ndarray:
    vals = np.array(
        [r[field] for r in records if group_fn(r) == group and r["method"] == method],
        dtype=float,
    )
    vals = vals[np.isfinite(vals)]
    return vals if vals.size else np.array([np.nan])


def finite_array_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def plot_metrics(records: list[dict], plot_dir: Path | None, show: bool) -> None:
    if not (plot_dir or show):
        return
    montages = sorted({r["montage_id"] for r in records}, key=montage_sort_key)
    montage_snr = sorted(
        {(r["montage_id"], r["snr_bin_db"]) for r in records},
        key=lambda x: (montage_sort_key(x[0]), x[1]),
    )
    plot_metric_grid(
        records,
        lambda r: r["montage_id"],
        montages,
        "Metrics by montage",
        plot_dir / "metrics_by_montage.png" if plot_dir else None,
        show,
    )
    plot_metric_grid(
        records,
        lambda r: (r["montage_id"], r["snr_bin_db"]),
        montage_snr,
        "Metrics by montage and 5 dB SNR bin",
        plot_dir / "metrics_by_montage_snr.png" if plot_dir else None,
        show,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate classic EEG inverse methods on synthgen data.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/validate_classic_fsaverage.yaml"))
    parser.add_argument("--forward-dir", type=Path, default=Path("banks/leadfield"))
    parser.add_argument("--methods", nargs="+", default=["sLORETA", "LCMV"])
    parser.add_argument("--n-samples", type=int, default=200, help="Numero massimo di sample da usare per ogni montaggio.")
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    config = GenerationConfig.from_yaml(str(args.config))
    datasets = [
        MontageDataset(path)
        for path in sorted(args.dataset_dir.iterdir())
        if path.is_dir() and (path / "metadata.jsonl").exists() and (path / "data.zarr").exists()
    ]
    if not datasets:
        raise FileNotFoundError(f"No dataset for montage found in {args.dataset_dir}")
    leadfield_bank = LeadfieldBank(config.leadfield_bank)
    print(f"Montages: {len(datasets)} | max samples/montage: {args.n_samples} | methods: {args.methods}")

    records = []
    for i, dataset in enumerate(datasets):
        records.extend(evaluate_dataset(dataset, args.n_samples, config, leadfield_bank, args.forward_dir, args.methods))
        print(f"Computed dataset {i+1}/{len(datasets)}")
    print_summary(records)
    plot_metrics(records, args.plot_dir, args.show_plots)


if __name__ == "__main__":
    main()
