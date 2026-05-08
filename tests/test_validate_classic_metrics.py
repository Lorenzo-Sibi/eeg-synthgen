import numpy as np

from scripts.validate_classic_methods import SurfaceGeometry, compute_paper_metrics


def test_compute_paper_metrics_uses_surface_areas_for_overlap():
    geometry = SurfaceGeometry(
        vertex_coords_mm=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        vertex_areas_mm2=np.array([2.0, 2.0, 2.0, 10.0], dtype=np.float32),
    )
    support = np.array([False, True, True, False])
    scores = np.array([0.1, 3.0, 2.0, 4.0])

    metrics = compute_paper_metrics(scores, support, geometry)

    assert metrics["gt_area_cm2"] == 0.04
    assert metrics["estimated_area_cm2"] == 0.14
    assert metrics["overlap_area_cm2"] == 0.04
    assert metrics["recall"] == 1.0
    assert np.isclose(metrics["precision"], 4.0 / 14.0)
    assert np.isfinite(metrics["le_mm"])
    assert np.isfinite(metrics["sd_mm"])
