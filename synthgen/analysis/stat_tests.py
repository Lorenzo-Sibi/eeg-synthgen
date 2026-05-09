from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, mannwhitneyu, tukey_hsd


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


def anova_tukey_per_metric(
    records: list[dict],
    metrics: list[str],
    group_key: str = "method",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """For each metric: pairwise Tukey-HSD across the levels of `group_key`.

    Returns a long-format DataFrame with columns:
        metric, group_a, group_b, mean_diff, p_value, reject_h0
    Empty DataFrame if there are fewer than 2 groups for ALL metrics.
    Falls back to one-way ANOVA p-value if Tukey fails on degenerate input.
    """
    out_rows: list[dict] = []
    for metric in metrics:
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in records:
            value = row.get(metric)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            buckets[row[group_key]].append(float(value))
        groups = sorted(buckets.keys())
        if len(groups) < 2:
            continue

        samples = [np.asarray(buckets[g]) for g in groups]
        try:
            tukey_res = tukey_hsd(*samples)
            pvalues = tukey_res.pvalue
        except Exception:
            f_stat, p_val = f_oneway(*samples)
            for a, b in combinations(range(len(groups)), 2):
                out_rows.append({
                    "metric": metric,
                    "group_a": groups[a],
                    "group_b": groups[b],
                    "mean_diff": float(np.mean(samples[a]) - np.mean(samples[b])),
                    "p_value": float(p_val),
                    "reject_h0": bool(p_val < alpha),
                })
            continue

        for a, b in combinations(range(len(groups)), 2):
            mean_diff = float(np.mean(samples[a]) - np.mean(samples[b]))
            p = float(pvalues[a, b])
            out_rows.append({
                "metric": metric,
                "group_a": groups[a],
                "group_b": groups[b],
                "mean_diff": mean_diff,
                "p_value": p,
                "reject_h0": bool(p < alpha),
            })

    return pd.DataFrame(
        out_rows,
        columns=["metric", "group_a", "group_b", "mean_diff", "p_value", "reject_h0"],
    )
