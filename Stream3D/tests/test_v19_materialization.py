from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from tools.diagnose_v19_materialization import classify_failure_type, _dilate_points


class V19MaterializationPureTest(unittest.TestCase):
    def test_failure_type_prefers_explicit_no_coverage(self) -> None:
        failure = classify_failure_type(
            gt_point_count=1000,
            num_surfels=0,
            mesh_recall=0.0,
            num_components=0,
            largest_component_ratio=0.0,
            best_component_recall=0.0,
            best_component_precision=0.0,
            exported_point_count=0,
        )
        self.assertEqual(failure, "no_surfel_coverage")

    def test_failure_type_detects_fragmentation_before_underfill(self) -> None:
        failure = classify_failure_type(
            gt_point_count=1000,
            num_surfels=80,
            mesh_recall=0.20,
            num_components=8,
            largest_component_ratio=0.30,
            best_component_recall=0.20,
            best_component_precision=0.95,
            exported_point_count=250,
        )
        self.assertEqual(failure, "fragmented")

    def test_dilate_points_keeps_seed_and_radius_neighbors(self) -> None:
        points = np.asarray(
            [
                [0.00, 0.00, 0.00],
                [0.02, 0.00, 0.00],
                [0.20, 0.00, 0.00],
            ],
            dtype=np.float32,
        )
        out = _dilate_points(points, cKDTree(points), {0}, 0.05)
        self.assertEqual(out, {0, 1})


if __name__ == "__main__":
    unittest.main()
