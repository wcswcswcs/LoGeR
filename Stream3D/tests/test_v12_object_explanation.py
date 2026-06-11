from __future__ import annotations

import unittest

import numpy as np

from stream4d.evidence_terms import ExplanationParams
from stream4d.measurement_bank import MeasurementBank
from stream4d.object_explanation import explain_objects


def _toy_bank() -> MeasurementBank:
    n = 8
    t = 3
    visible = np.ones((t, n), dtype=bool)
    target = np.zeros((t, n), dtype=np.int64)
    target[:, :6] = 7
    target[1, 6:] = 0
    return MeasurementBank(
        scene="toy",
        frame_ids=np.asarray([0, 1, 2], dtype=np.int64),
        carrier_id=np.arange(n, dtype=np.int64),
        uv_pred=np.full((t, n, 2), 0.5, dtype=np.float32),
        valid=visible.copy(),
        visibility=np.ones((t, n), dtype=np.float32),
        confidence=np.ones((t, n), dtype=np.float32),
        src_frame_global=np.zeros((n,), dtype=np.int64),
        src_mask_id=np.asarray([7, 7, 7, 7, 7, 7, 0, 0], dtype=np.int64),
        src_xy=np.zeros((n, 2), dtype=np.int64),
        src_rgb=np.ones((n, 3), dtype=np.float32) * 0.5,
        target_mask_id=target,
        target_in_bounds=visible.copy(),
        visible_ok=visible.copy(),
        boundary_distance=np.ones((t, n), dtype=np.float32) * 5.0,
        source_boundary_distance=np.ones((n,), dtype=np.float32) * 5.0,
        mask_frame_available=np.ones((t,), dtype=bool),
        positive_observation=target > 0,
        negative_observation=np.zeros((t, n), dtype=bool),
        source_positive_propagated=visible & (np.asarray([7, 7, 7, 7, 7, 7, 0, 0], dtype=np.int64)[None, :] > 0),
        meta={},
    )


class V12ObjectExplanationTest(unittest.TestCase):
    def test_birth_and_export_observations(self) -> None:
        params = ExplanationParams(
            birth_min_surfels=4,
            min_core_surfels_per_object=4,
            birth_min_boundary_safe_ratio=0.5,
            birth_max_ambiguous_ratio=1.0,
            measurement_min_surfels=2,
            core_posterior_threshold=0.4,
        )
        slots, diag = explain_objects(_toy_bank(), params=params, mode="with_negative", seed=1)
        self.assertEqual(len(slots), 1)
        self.assertGreaterEqual(slots[0].core_surfels.shape[0], 4)
        self.assertTrue(slots[0].mask_observations)
        self.assertEqual(diag["same_frame_cannot_link_violations"], 0)


if __name__ == "__main__":
    unittest.main()
