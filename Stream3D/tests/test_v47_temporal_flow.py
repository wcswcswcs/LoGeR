from __future__ import annotations

import unittest

from stream4d_native.v47_object_field_export import export_object_fields_from_tracklets
from stream4d_native.v47_temporal_edge_builder import build_temporal_candidate_edges
from stream4d_native.v47_tracklet_builder import build_tracklets
from stream4d_native.v47_underseg_shared_observation import shared_observation_rows


class V47TemporalFlowTest(unittest.TestCase):
    def test_edge_builder_uses_d4rt_temporal_containment_without_gt_prediction(self) -> None:
        mask_rows = [
            {
                "node_id": 1,
                "mask_observation_id": "s:0:1",
                "scene": "s",
                "frame_id": 0,
                "mask_id": 1,
                "bbox_x0": 0,
                "bbox_y0": 0,
                "bbox_x1": 9,
                "bbox_y1": 9,
                "core_feature": [1.0, 0.0],
                "diagnostic_gt_instance": 7,
            },
            {
                "node_id": 2,
                "mask_observation_id": "s:1:2",
                "scene": "s",
                "frame_id": 1,
                "mask_id": 2,
                "bbox_x0": 1,
                "bbox_y0": 0,
                "bbox_x1": 10,
                "bbox_y1": 9,
                "core_feature": [1.0, 0.0],
                "diagnostic_gt_instance": 7,
            },
        ]
        carrier_rows = [
            {"scene": "s", "frame_id": 0, "carrier_id": 11, "visible": True, "valid_uv": True, "observed_mask_id": 1},
            {"scene": "s", "frame_id": 0, "carrier_id": 12, "visible": True, "valid_uv": True, "observed_mask_id": 1},
            {"scene": "s", "frame_id": 1, "carrier_id": 11, "visible": True, "valid_uv": True, "observed_mask_id": 2},
            {"scene": "s", "frame_id": 1, "carrier_id": 12, "visible": True, "valid_uv": True, "observed_mask_id": 2},
        ]
        payload = build_temporal_candidate_edges(mask_rows=mask_rows, carrier_rows=carrier_rows)
        self.assertEqual(len(payload["edge_rows"]), 1)
        edge = payload["edge_rows"][0]
        self.assertAlmostEqual(edge["d4rt_forward_containment"], 1.0)
        self.assertTrue(edge["diagnostic_same_gt"])
        self.assertFalse(edge["uses_gt_for_prediction"])

    def test_tracklet_export_has_mask_evidence_and_no_d4rt_birth(self) -> None:
        mask_rows = [
            {"node_id": 1, "scene": "s", "frame_id": 0, "mask_id": 1, "diagnostic_gt_instance": 7},
            {"node_id": 2, "scene": "s", "frame_id": 1, "mask_id": 2, "diagnostic_gt_instance": 7},
        ]
        edge_rows = [
            {
                "src_node_id": 1,
                "dst_node_id": 2,
                "scene": "s",
                "edge_type": "adjacent",
                "A5_d4rt_semantic_confirmation": 0.9,
                "edge_cost": 0.1,
                "edge_accept_candidate": True,
            }
        ]
        payload = build_tracklets(mask_rows=mask_rows, edge_rows=edge_rows, min_score=0.3)
        fields = export_object_fields_from_tracklets(payload["tracklet_rows"])
        self.assertEqual(len(fields), 1)
        self.assertTrue(fields[0]["has_mask_or_mask_atom_evidence"])
        self.assertFalse(fields[0]["birth_from_d4rt_tube"])

    def test_shared_observation_cannot_create_identity_merge_edge(self) -> None:
        carrier_rows = [
            {"scene": "s", "frame_id": 0, "carrier_id": 1, "observed_mask_id": 4},
            {"scene": "s", "frame_id": 0, "carrier_id": 2, "observed_mask_id": 4},
        ]
        rows = shared_observation_rows(carrier_rows, {("s", 1): "t_a", ("s", 2): "t_b"})
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["shared_observation"])
        self.assertFalse(rows[0]["can_create_identity_merge_edge"])


if __name__ == "__main__":
    unittest.main()

