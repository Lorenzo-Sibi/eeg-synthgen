from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import kruskal, mannwhitneyu


def kruskal_wallis_test(groups):
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    k, n = len(arrays), sum(len(a) for a in arrays)
    H, p = kruskal(*arrays)
    eta2 = float(np.clip((H - k + 1) / max(n - k, 1), 0.0, 1.0))
    return {"H": float(H), "p_value": float(p), "eta_squared": eta2, "k": k, "n": int(n)}


def pairwise_mannwhitney_holm(groups, group_names):
    """Pairwise Mann-Whitney U with Holm-Bonferroni FWER control.

    Not Dunn's test (which uses joint ranks). Holm 1979 dominates plain
    Bonferroni at the same FWER.
    """
    pairs = list(combinations(range(len(groups)), 2))
    raw = np.array([
        mannwhitneyu(groups[i], groups[j], alternative="two-sided").pvalue
        for i, j in pairs
    ])
    m = raw.size
    order = np.argsort(raw)
    adj = np.empty(m)
    cummax = 0.0
    for rank, idx in enumerate(order):
        cummax = max(cummax, (m - rank) * raw[idx])
        adj[idx] = min(cummax, 1.0)
    return {
        f"{group_names[i]}_vs_{group_names[j]}": {
            "p_raw": float(raw[k]),
            "p_holm": float(adj[k]),
        }
        for k, (i, j) in enumerate(pairs)
    }


def intraclass_correlation(values_by_group):
    """ICC(1,1) one-way random effects (Shrout & Fleiss 1979)."""
    arrays = [np.asarray(v, dtype=np.float64) for v in values_by_group.values()]
    k = len(arrays)
    ns = np.array([len(a) for a in arrays], dtype=float)
    n_total = float(ns.sum())
    grand = np.concatenate(arrays).mean()
    ss_b = sum(n * (a.mean() - grand) ** 2 for a, n in zip(arrays, ns))
    ss_w = sum(np.sum((a - a.mean()) ** 2) for a in arrays)
    df_b, df_w = max(k - 1, 1), max(n_total - k, 1)
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    n0 = (n_total - float(np.sum(ns ** 2) / n_total)) / df_b
    denom = ms_b + max(n0 - 1, 0) * ms_w
    if denom <= 1e-30:
        return 1.0
    return float(np.clip((ms_b - ms_w) / denom, -1.0, 1.0))
