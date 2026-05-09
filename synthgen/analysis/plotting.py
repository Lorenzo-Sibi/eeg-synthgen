"""Publication-quality boxplot helpers for inverse-method validation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_METHOD_ORDER = ["MNE", "dSPM", "sLORETA", "eLORETA", "LCMV"]

METRIC_LABELS = {
    "le_mm":   "LE [mm] ↓",
    "te_ms":   "TE [ms] ↓",
    "nmse":    "nMSE ↓",
    "psnr_db": "PSNR [dB] ↑",
    "auc":     "AUC ↑",
}


def _records_to_long_df(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def _filter_methods(df: pd.DataFrame, method_order: list[str] | None) -> list[str]:
    available = list(df["method"].unique())
    if method_order is None:
        return [m for m in DEFAULT_METHOD_ORDER if m in available] + \
               [m for m in available if m not in DEFAULT_METHOD_ORDER]
    return [m for m in method_order if m in available]


def _make_boxplot_axis(ax, df, x_col, x_order, methods, metric):
    """Draw side-by-side boxes for each method within each x category."""
    cmap = plt.get_cmap("tab10")
    method_colors = {m: cmap(i % 10) for i, m in enumerate(methods)}
    n_methods = len(methods)
    box_width = 0.8 / max(n_methods, 1)

    for x_pos, x_val in enumerate(x_order):
        for j, method in enumerate(methods):
            mask = (df[x_col] == x_val) & (df["method"] == method)
            values = df.loc[mask, metric].dropna().to_numpy()
            if len(values) == 0:
                continue
            offset = (j - (n_methods - 1) / 2) * box_width
            ax.boxplot(
                [values],
                positions=[x_pos + offset],
                widths=box_width * 0.9,
                whis=1.5,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="o", markersize=3, markerfacecolor="lightgrey",
                                markeredgecolor="lightgrey", alpha=0.4),
                medianprops=dict(color="black", linewidth=1.2),
                boxprops=dict(facecolor=method_colors[method], alpha=0.7,
                              edgecolor="black", linewidth=0.8),
                whiskerprops=dict(color="black", linewidth=0.8),
                capprops=dict(color="black", linewidth=0.8),
            )
            ax.scatter([x_pos + offset], [values.mean()], marker="D",
                       s=20, color="white", edgecolor="black", linewidth=0.6, zorder=5)

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(
        [str(x) for x in x_order],
        rotation=30 if len(x_order) > 3 else 0,
        ha="right" if len(x_order) > 3 else "center",
    )
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=method_colors[m], alpha=0.7, edgecolor="black")
               for m in methods]
    ax.legend(handles, methods, loc="best", fontsize=8, framealpha=0.9)


def boxplot_per_montage(
    records: list[dict],
    metrics: list[str],
    output_dir: Path,
    *,
    method_order: list[str] | None = None,
    montage_order: list[str] | None = None,
) -> None:
    """1×N subplot figure: x=montage_id, hue=method, one subplot per metric."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _records_to_long_df(records)
    methods = _filter_methods(df, method_order)
    montages = montage_order or sorted(df["montage_id"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        _make_boxplot_axis(ax, df, "montage_id", montages, methods, metric)
        ax.set_xlabel("Montage")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_metrics_per_montage.pdf")
    fig.savefig(output_dir / "fig_metrics_per_montage.png", dpi=300)
    plt.close(fig)


def boxplot_per_snr_bin(
    records: list[dict],
    metrics: list[str],
    output_dir: Path,
    *,
    method_order: list[str] | None = None,
    snr_bins_db: list[int] | None = None,
) -> None:
    """1×N subplot figure: x=snr_bin_db, hue=method, one subplot per metric."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _records_to_long_df(records)
    methods = _filter_methods(df, method_order)
    bins = snr_bins_db or sorted(df["snr_bin_db"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        _make_boxplot_axis(ax, df, "snr_bin_db", bins, methods, metric)
        ax.set_xlabel("SNR bin [dB]")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_metrics_per_snr.pdf")
    fig.savefig(output_dir / "fig_metrics_per_snr.png", dpi=300)
    plt.close(fig)
