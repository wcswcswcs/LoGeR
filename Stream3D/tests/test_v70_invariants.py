from __future__ import annotations

import unittest

from stream4d_native.v70_masklet_tracklets import _tracklet_score
from stream4d_native.v70_object_capsule import _build_capsule_mapping
from stream4d_native.v70_true_material_closure import _variant_score


class V70InvariantTests(unittest.TestCase):
    def _closure_row(self, **overrides):
        row = {
            "_inside_ratio": 0.9,
            "_outside_ratio": 0.1,
            "_candidate_underseg_risk": False,
            "frame_delta": 5,
            "scene_id": "scene",
            "chunk_id": "scene:chunk000",
            "anchor_frame": 0,
            "anchor_mask": 1,
            "candidate_frame": 5,
            "candidate_mask": 2,
            "anchor_DINO_mode_id": "",
            "candidate_DINO_mode_id": "",
            "anchor_repeated_signature_id": "sig",
            "candidate_repeated_signature_id": "sig",
        }
        row.update(overrides)
        return row

    def _tracklet_edge(self, **overrides):
        edge = {
            "inside_ratio": 0.9,
            "outside_ratio": 0.1,
            "residual": 0.8,
            "frame_delta": 5,
            "underseg": False,
            "dino": 0.0,
            "signature": 1.0,
            "temporal": 1.0,
            "shuffle": 0.0,
        }
        edge.update(overrides)
        return edge

    def test_true_closure_underseg_is_shared_or_rejected(self):
        score, role = _variant_score("TC5_TC4_underseg_shared", self._closure_row(_candidate_underseg_risk=True))
        self.assertGreater(score, 0.0)
        self.assertIn(role, {"shared", "reject"})
        self.assertNotEqual(role, "core")

    def test_true_closure_no_temporal_control_is_not_temporal_variant(self):
        row = self._closure_row(frame_delta=60)
        temporal_score, temporal_role = _variant_score("TC6_TC5_temporal_adjacency", row)
        control_score, control_role = _variant_score("TC8_no_temporal_carrier_control", row)
        self.assertIn(temporal_role, {"core", "support", "shared", "reject"})
        self.assertIn(control_role, {"core", "support", "shared", "reject"})
        self.assertNotEqual(temporal_score, control_score)

    def test_tracklet_underseg_is_not_core_bridge(self):
        score, role = _tracklet_score("TR4_TR3_underseg_shared", self._tracklet_edge(underseg=True))
        self.assertGreater(score, 0.0)
        self.assertIn(role, {"shared", "reject"})
        self.assertNotEqual(role, "core")

    def test_object_capsule_shared_node_is_not_core_bridge(self):
        shared = (5, 2)
        ordinary = (10, 3)
        mapping = _build_capsule_mapping(
            variant="OC4_shared_ledger_carrier_veto_t055",
            nodes={(0, 1), shared, ordinary},
            shared_nodes={shared},
            edges=[
                {"left": (0, 1), "right": shared, "score": 0.95},
                {"left": (0, 1), "right": ordinary, "score": 0.60},
            ],
        )
        self.assertNotIn(shared, mapping)
        self.assertEqual(mapping[(0, 1)], mapping[ordinary])

    def test_shared_anchor_coref_uses_shared_as_evidence_only(self):
        shared = (20, 9)
        mapping = _build_capsule_mapping(
            variant="OC11_shared_anchor_coref_t055_b050",
            nodes={(0, 1), (10, 2), (5, 3), (15, 4), shared},
            shared_nodes={shared},
            edges=[
                {"left": (0, 1), "right": (10, 2), "score": 0.70},
                {"left": (5, 3), "right": (15, 4), "score": 0.70},
                {"left": shared, "right": (0, 1), "score": 0.65},
                {"left": shared, "right": (5, 3), "score": 0.64},
            ],
        )
        self.assertNotIn(shared, mapping)
        self.assertEqual(mapping[(0, 1)], mapping[(5, 3)])
        self.assertEqual(mapping[(10, 2)], mapping[(15, 4)])


if __name__ == "__main__":
    unittest.main()
