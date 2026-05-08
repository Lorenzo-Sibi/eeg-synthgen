from __future__ import annotations

import numpy as np


def summarize_spatial_coverage(active_vertex_sets, n_vertices):
    counts = np.zeros(n_vertices, dtype=np.float64)
    for indices in active_vertex_sets:
        counts[np.asarray(indices, dtype=np.int64)] += 1
    total = counts.sum()
    n_unique = int((counts > 0).sum())
    max_entropy = float(np.log2(max(n_vertices, 2)))
    if total <= 0:
        return dict(
            coverage_fraction=0.0,
            entropy_bits=0.0,
            max_entropy_bits=max_entropy,
            uniformity_index=0.0,
            n_unique_vertices=0,
        )
    probs = counts[counts > 0] / total
    
    entropy = float(-np.sum(probs * np.log2(probs)))
    return dict(
        coverage_fraction=n_unique / max(n_vertices, 1),
        entropy_bits=entropy,
        max_entropy_bits=max_entropy,
        uniformity_index=entropy / max_entropy if max_entropy > 0 else 0.0,
        n_unique_vertices=n_unique,
    )
