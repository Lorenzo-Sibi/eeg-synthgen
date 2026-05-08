import numpy as np
import pytest

from synthgen.analysis.spatial import summarize_spatial_coverage


def test_full_activation():
    r = summarize_spatial_coverage([np.arange(50)], 50)
    assert r["coverage_fraction"] == pytest.approx(1.0)
    assert r["uniformity_index"] == pytest.approx(1.0, rel=1e-3)


def test_single_vertex():
    r = summarize_spatial_coverage([np.array([3])] * 20, 100)
    assert r["entropy_bits"] == pytest.approx(0.0, abs=1e-9)
    assert r["coverage_fraction"] == pytest.approx(0.01)


def test_uniform_max_entropy():
    n = 16
    r = summarize_spatial_coverage([np.array([i]) for i in range(n)], n)
    assert r["entropy_bits"] == pytest.approx(np.log2(n), rel=1e-3)


def test_returns_required_keys():
    r = summarize_spatial_coverage([np.array([0])], 10)
    assert {
        "coverage_fraction",
        "entropy_bits",
        "max_entropy_bits",
        "uniformity_index",
        "n_unique_vertices",
    }.issubset(r)


def test_empty_input():
    r = summarize_spatial_coverage([], 10)
    assert r["coverage_fraction"] == 0.0
    assert r["entropy_bits"] == 0.0
    assert r["n_unique_vertices"] == 0
