#!/usr/bin/env python3
"""Visual diagnostic for a projected connectivity bank.

Renders three panels:

  1. Mapping plot — base-connectome centroids (large numbered spheres) plus
     target-parcel centroids (small spheres) coloured by their nearest-centroid
     assignment. Lines connect each target to its assigned base. A red line
     flags any assignment whose distance exceeds ``--max-dist-mm`` (default
     50 mm) — those are the candidates for misalignment / out-of-coverage
     parcels. Both centroid clouds are translated to zero-mean before plotting
     and before distances are reported, matching what
     ``_compute_target_to_base`` does internally; otherwise the bulk offset
     between TVB-format frame and MNE head frame (~50 mm in z for fsaverage)
     dominates the visualisation and makes a correct mapping look broken.

  2. Heatmap of the projected weight matrix W (target x target), rows/cols
     ordered by hemisphere then by anatomical label. Diagonal LH-LH and
     RH-RH blocks should dominate; cross-hemispheric structure should be
     visible (callosal connections). Annotated with key matrix statistics
     (max row sum, density, asymmetry).

  3. Top-K strongest edges in 3D, drawn between target centroids over the
     anatomy's source-space cloud. Edge thickness encodes weight. Confirms
     that the strongest connections are anatomically plausible (short-range
     intra-hemispheric or cross-callosal long-range).

Usage:
    python scripts/visualize_connectivity.py \\
        --config config/default.yaml \\
        --anatomy fsaverage --scheme desikan_killiany --tvb-base 192 \\
        --output banks/connectivity/desikan_killiany.png

Add ``--show`` for an interactive window. Add ``--top-edges 80`` to draw more
edges in panel 3.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from synthgen.banks.anatomy import AnatomyBank
from synthgen.banks.connectivity import ConnectivityBank
from synthgen.config import GenerationConfig


_TVB_BASE_CHOICES = (66, 68, 76, 96, 192)


def _load_tvb_base(n_regions: int):
    import tvb_data.connectivity as tc
    from tvb.datatypes.connectivity import Connectivity as TVBConnectivity

    if n_regions not in _TVB_BASE_CHOICES:
        raise ValueError(f"--tvb-base must be one of {_TVB_BASE_CHOICES}")
    zip_path = os.path.join(os.path.dirname(tc.__file__), f"connectivity_{n_regions}.zip")
    return TVBConnectivity.from_file(zip_path), f"tvb_{n_regions}"


def _load_deepsif_base(deepsif_dir: Path):
    from tvb.datatypes.connectivity import Connectivity as TVBConnectivity
    zip_path = (deepsif_dir / "connectivity_998.zip").resolve()
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Run scripts/fetch_atlases.py --deepsif"
        )
    return TVBConnectivity.from_file(str(zip_path)), "deepsif_998"


def _hemisphere_of_label(label: str) -> str:
    """DK / Destrieux labels end with -lh / -rh; fall back to 'lh' if unknown."""
    if label.endswith("-lh") or label.startswith("lh."):
        return "lh"
    if label.endswith("-rh") or label.startswith("rh."):
        return "rh"
    return "lh"


def _panel_mapping(ax, base_centers, target_centers, target_to_base, distances, max_dist_mm):
    """3D scatter: base centroids + target centroids coloured by assignment."""
    import matplotlib.pyplot as plt
    n_base = len(base_centers)
    cmap = plt.get_cmap("tab20", max(n_base, 20))
    colors = cmap(np.arange(n_base) % cmap.N)

    # Base centroids (large, numbered)
    for b, c in enumerate(base_centers):
        ax.scatter(c[0], c[1], c[2], s=120, color=colors[b], edgecolor="black", zorder=5)
        ax.text(c[0], c[1], c[2], f"{b}", fontsize=6, ha="center", va="center", zorder=6)

    # Target centroids + line to assigned base
    bad = 0
    for i, (tc_i, b) in enumerate(zip(target_centers, target_to_base)):
        ax.scatter(tc_i[0], tc_i[1], tc_i[2], s=20, color=colors[b], alpha=0.85, zorder=4)
        line_color = "red" if distances[i] > max_dist_mm else colors[b]
        line_alpha = 0.8 if distances[i] > max_dist_mm else 0.2
        ax.plot(
            [tc_i[0], base_centers[b, 0]],
            [tc_i[1], base_centers[b, 1]],
            [tc_i[2], base_centers[b, 2]],
            color=line_color,
            alpha=line_alpha,
            linewidth=1.2 if distances[i] > max_dist_mm else 0.5,
            zorder=3,
        )
        if distances[i] > max_dist_mm:
            bad += 1

    ax.set_title(
        f"Mapping (base={n_base}, target={len(target_centers)}, zero-mean centered)\n"
        f"mean dist {distances.mean():.1f} mm, max {distances.max():.1f} mm, "
        f"flagged > {max_dist_mm} mm: {bad}",
        fontsize=8,
    )
    ax.set_xlabel("x (mm)", fontsize=7)
    ax.set_ylabel("y (mm)", fontsize=7)
    ax.set_zlabel("z (mm)", fontsize=7)
    ax.tick_params(labelsize=6)


def _panel_heatmap(ax, W, labels):
    """Reorder rows/cols by hemisphere then label, plot heatmap with stats."""
    hemi = np.array([_hemisphere_of_label(label) for label in labels])
    order = np.lexsort((np.array(labels), hemi))
    W_ord = W[np.ix_(order, order)]
    labels_ord = [labels[i] for i in order]
    hemi_ord = hemi[order]

    im = ax.imshow(W_ord, cmap="magma", aspect="equal")
    n_lh = int((hemi_ord == "lh").sum())
    R = W.shape[0]

    # Hemisphere divider
    ax.axhline(n_lh - 0.5, color="cyan", linewidth=0.7)
    ax.axvline(n_lh - 0.5, color="cyan", linewidth=0.7)

    asym = float(np.linalg.norm(W - W.T) / (np.linalg.norm(W) + 1e-30))
    density = float((W > 0).mean()) * 100.0
    row_sum_max = float(W.sum(axis=1).max())
    p50, p90, p99 = np.percentile(W[W > 0], [50, 90, 99]) if (W > 0).any() else (0, 0, 0)

    ax.set_title(
        f"W ({R}x{R})  density {density:.1f}%  max(row_sum) {row_sum_max:.3f}\n"
        f"asym ||W-W^T||/||W|| {asym:.2e}  W>0 percentiles: "
        f"50={p50:.3f}, 90={p90:.3f}, 99={p99:.3f}",
        fontsize=8,
    )
    ax.set_xlabel("target region (sorted: lh | rh)", fontsize=7)
    ax.set_ylabel("target region", fontsize=7)
    if R <= 80:
        ax.set_xticks(range(R))
        ax.set_yticks(range(R))
        ax.set_xticklabels(labels_ord, rotation=90, fontsize=4)
        ax.set_yticklabels(labels_ord, fontsize=4)
    else:
        ax.tick_params(labelsize=5)
    import matplotlib.pyplot as plt
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _panel_edges(ax, W, region_centers, source_space, top_k, labels):
    """Render top-K strongest edges in 3D over the anatomy's source-space."""
    R = W.shape[0]
    iu = np.triu_indices(R, k=1)
    pairs_w = W[iu]
    if pairs_w.size == 0 or pairs_w.max() == 0:
        ax.set_title("Top edges (none — W is zero)", fontsize=8)
        return
    k = min(top_k, int((pairs_w > 0).sum()))
    top_idx = np.argsort(pairs_w)[-k:]
    rows = iu[0][top_idx]
    cols = iu[1][top_idx]
    weights = pairs_w[top_idx]
    w_max = float(weights.max())

    # Anatomy cloud (subsampled for speed)
    if source_space is not None:
        vc = source_space.vertex_coords
        n_show = min(5000, vc.shape[0])
        idx = np.random.default_rng(0).choice(vc.shape[0], n_show, replace=False)
        ax.scatter(vc[idx, 0], vc[idx, 1], vc[idx, 2], s=0.3, alpha=0.05, c="grey")

    # Edges
    for i, j, w in zip(rows, cols, weights):
        a, b = region_centers[i], region_centers[j]
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
            color="crimson",
            alpha=0.25 + 0.7 * (w / w_max),
            linewidth=0.3 + 2.5 * (w / w_max),
        )

    ax.scatter(
        region_centers[:, 0], region_centers[:, 1], region_centers[:, 2],
        s=25, c="navy", edgecolor="white", linewidth=0.4, zorder=5,
    )
    ax.set_title(
        f"Top-{k} edges (max W={w_max:.3f}, min shown={weights.min():.3f})",
        fontsize=8,
    )
    ax.set_xlabel("x (mm)", fontsize=7)
    ax.set_ylabel("y (mm)", fontsize=7)
    ax.set_zlabel("z (mm)", fontsize=7)
    ax.tick_params(labelsize=6)


def _region_centers_from_anatomy(source_space, n_regions: int) -> np.ndarray:
    parc = source_space.parcellation
    centers = np.zeros((n_regions, 3), dtype=np.float32)
    for r in range(n_regions):
        m = parc == r
        if m.any():
            centers[r] = source_space.vertex_coords[m].mean(axis=0)
    return centers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--anatomy", type=str, default="fsaverage")
    ap.add_argument("--scheme", type=str, default="desikan_killiany")
    ap.add_argument("--tvb-base", type=int, default=192, choices=_TVB_BASE_CHOICES,
                    help="TVB bundled connectome to use as base when --base=tvb (default 192).")
    ap.add_argument("--base", type=str, default="auto", choices=("auto", "tvb", "deepsif"),
                    help="Which base connectome to compare against. 'auto' uses deepsif if "
                         "the scheme is deepsif_994 and a connectivity_998.zip is found, else "
                         "TVB. (default: auto)")
    ap.add_argument("--deepsif-dir", type=Path, default=Path("banks/atlases/deepsif"),
                    help="Directory containing connectivity_998.zip (used when --base=deepsif).")
    ap.add_argument("--top-edges", type=int, default=40)
    ap.add_argument("--max-dist-mm", type=float, default=50.0,
                    help="Highlight target->base assignments above this distance (default 50mm).")
    ap.add_argument("--output", type=Path, default=None,
                    help="PNG output (default: banks/connectivity/{scheme}.png).")
    ap.add_argument("--show", action="store_true",
                    help="Open the interactive window in addition to saving the file.")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = GenerationConfig.from_yaml(args.config)
    anat = AnatomyBank(cfg.anatomy_bank).load(args.anatomy, scheme=args.scheme)
    conn = ConnectivityBank(cfg.connectivity_bank).load(args.scheme)
    if conn.scheme != args.scheme:
        raise ValueError(f"bank scheme {conn.scheme!r} != requested {args.scheme!r}")

    base_kind = args.base
    if base_kind == "auto":
        if args.scheme == "deepsif_994" and (args.deepsif_dir / "connectivity_998.zip").exists():
            base_kind = "deepsif"
        else:
            base_kind = "tvb"

    if base_kind == "deepsif":
        base, base_tag = _load_deepsif_base(args.deepsif_dir)
    else:
        base, base_tag = _load_tvb_base(args.tvb_base)
    base_centers = np.asarray(base.centres, dtype=np.float32)
    target_centers = _region_centers_from_anatomy(anat, conn.weights.shape[0])

    # Reuse the same mapping logic that prepare_connectivity.py applied when
    # the bank was built. scripts/ is not a package, so load by file path.
    import importlib.util
    pc_path = Path(__file__).parent / "prepare_connectivity.py"
    spec = importlib.util.spec_from_file_location("prepare_connectivity", pc_path)
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    target_to_base = pc._compute_target_to_base(base_centers, target_centers)
    # Match the centering convention used by _compute_target_to_base so that
    # the reported distance reflects matching quality (not the bulk offset
    # between the base connectome's frame and the anatomy's head frame).
    base_plot = base_centers - base_centers.mean(axis=0)
    target_plot = target_centers - target_centers.mean(axis=0)
    distances = np.linalg.norm(
        base_plot[target_to_base] - target_plot, axis=1
    )

    fig = plt.figure(figsize=(20, 7))
    fig.suptitle(
        f"Connectivity diagnostic — {args.scheme} on {args.anatomy} (base={base_tag})",
        fontsize=11,
    )
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    _panel_mapping(ax1, base_plot, target_plot, target_to_base, distances, args.max_dist_mm)
    _panel_heatmap(ax2, conn.weights, conn.region_labels)
    _panel_edges(ax3, conn.weights, target_centers, anat, args.top_edges, conn.region_labels)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = args.output
    if out is None:
        out = Path(cfg.connectivity_bank.bank_dir) / f"{args.scheme}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()