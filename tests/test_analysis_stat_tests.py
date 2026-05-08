import numpy as np
import pytest

from synthgen.analysis.stat_tests import (
    intraclass_correlation,
    kruskal_wallis_test,
    pairwise_mannwhitney_holm,
)


def test_kruskal_wallis_separates_groups():
    rng = np.random.default_rng(0)
    g = [rng.normal(loc, 0.1, 50) for loc in (0, 5, 10)]
    r = kruskal_wallis_test(g)
    assert r["p_value"] < 0.001 and r["eta_squared"] > 0.5


def test_kruskal_wallis_identical():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 50)
    assert (
        kruskal_wallis_test([base, base.copy(), base.copy()])["p_value"] > 0.05
    )


def test_pairwise_mwu_holm_returns_all_pairs():
    rng = np.random.default_rng(2)
    groups = [rng.normal(i, 0.5, 30) for i in range(4)]
    r = pairwise_mannwhitney_holm(groups, ["a", "b", "c", "d"])
    assert len(r) == 6
    for v in r.values():
        assert v["p_holm"] >= v["p_raw"]


def test_holm_monotone_and_no_worse_than_bonferroni():
    rng = np.random.default_rng(7)
    groups = [rng.normal(0, 1, 50) for _ in range(5)]
    r = pairwise_mannwhitney_holm(groups, list("abcde"))
    sorted_holm = sorted(v["p_holm"] for v in r.values())
    assert sorted_holm == sorted(sorted_holm)
    m = len(r)
    for v in r.values():
        assert v["p_holm"] <= min(m * v["p_raw"], 1.0) + 1e-12


@pytest.mark.parametrize(
    "seed,sigma,expected_high",
    [(3, 0.01, True), (4, 1.0, False)],
)
def test_icc_separates(seed, sigma, expected_high):
    r = np.random.default_rng(seed)
    groups = {
        str(i): list(r.normal(i if expected_high else 0, sigma, 30))
        for i in range(5)
    }
    icc = intraclass_correlation(groups)
    assert (icc > 0.9) if expected_high else (icc < 0.3)
