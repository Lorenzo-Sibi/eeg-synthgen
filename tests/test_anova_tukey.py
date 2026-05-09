import numpy as np
import pandas as pd

from synthgen.analysis.stat_tests import anova_tukey_per_metric


def _make_records(n_per_group: int = 30, seed: int = 0):
    """Three groups with shifted means for metric 'le_mm'."""
    rng = np.random.RandomState(seed)
    rows = []
    for shift, method in [(0.0, "MNE"), (5.0, "sLORETA"), (5.0, "eLORETA")]:
        for _ in range(n_per_group):
            rows.append({"method": method, "le_mm": rng.randn() + shift})
    return rows


def test_anova_finds_known_significant_difference():
    records = _make_records()
    out = anova_tukey_per_metric(records, metrics=["le_mm"])
    assert isinstance(out, pd.DataFrame)
    pair = out[(out["group_a"] == "MNE") & (out["group_b"] == "sLORETA")]
    assert not pair.empty
    assert pair.iloc[0]["p_value"] < 0.01
    assert bool(pair.iloc[0]["reject_h0"])


def test_returns_pvalues_for_all_pairs():
    records = _make_records()
    out = anova_tukey_per_metric(records, metrics=["le_mm"])
    # 3 methods → C(3, 2) = 3 pairs
    assert len(out) == 3


def test_handles_single_group_gracefully():
    rows = [{"method": "MNE", "le_mm": x} for x in np.random.randn(20)]
    out = anova_tukey_per_metric(rows, metrics=["le_mm"])
    assert out.empty   # no pairs possible with one group
