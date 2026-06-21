from __future__ import annotations

import unittest

from stream4d_native.v55_chunk_roles import infer_chunk_roles_from_features


class V55ChunkRolesTest(unittest.TestCase):
    def test_role_prediction_uses_observation_features_not_diagnostic_local_quality(self) -> None:
        features = [
            {
                "scene": "s",
                "chunk_id": "s:chunk000",
                "chunk_index": 0,
                "mask_measurement_frame_count": 10,
                "mask_count": 120,
                "component_count": 100,
                "component_coverage_by_masks": 0.90,
                "representative_coverage": 0.80,
                "reprojection_success_rate": 0.90,
                "boundary_overlap_prev": 0.0,
                "boundary_overlap_next": 0.25,
                "scale_guard_status": True,
                "anchor_local_completeness": 0.10,
            },
            {
                "scene": "s",
                "chunk_id": "s:chunk001",
                "chunk_index": 1,
                "mask_measurement_frame_count": 2,
                "mask_count": 10,
                "component_count": 100,
                "component_coverage_by_masks": 0.85,
                "representative_coverage": 0.70,
                "reprojection_success_rate": 0.90,
                "boundary_overlap_prev": 0.25,
                "boundary_overlap_next": 0.25,
                "scale_guard_status": True,
                "anchor_local_completeness": 0.99,
            },
        ]
        rows, repairs = infer_chunk_roles_from_features(features, anchor_threshold=0.62)
        by_chunk = {row["chunk_id"]: row["role"] for row in rows}
        self.assertEqual(repairs, [])
        self.assertEqual(by_chunk["s:chunk000"], "anchor")
        self.assertNotEqual(by_chunk["s:chunk001"], "anchor")

    def test_no_anchor_repair_promotes_best_mask_supported_chunk(self) -> None:
        features = [
            {
                "scene": "s",
                "chunk_id": "s:chunk000",
                "chunk_index": 0,
                "mask_measurement_frame_count": 3,
                "mask_count": 20,
                "component_count": 100,
                "component_coverage_by_masks": 0.75,
                "representative_coverage": 0.55,
                "reprojection_success_rate": 0.70,
                "boundary_overlap_prev": 0.0,
                "boundary_overlap_next": 0.10,
                "scale_guard_status": True,
                "anchor_local_completeness": 0.1,
            },
            {
                "scene": "s",
                "chunk_id": "s:chunk001",
                "chunk_index": 1,
                "mask_measurement_frame_count": 2,
                "mask_count": 12,
                "component_count": 100,
                "component_coverage_by_masks": 0.72,
                "representative_coverage": 0.52,
                "reprojection_success_rate": 0.70,
                "boundary_overlap_prev": 0.10,
                "boundary_overlap_next": 0.0,
                "scale_guard_status": True,
                "anchor_local_completeness": 0.2,
            },
        ]
        rows, repairs = infer_chunk_roles_from_features(features, anchor_threshold=0.95)
        self.assertTrue(repairs)
        self.assertEqual(sum(1 for row in rows if row["role"] == "anchor"), 1)


if __name__ == "__main__":
    unittest.main()
