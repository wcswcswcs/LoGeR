from __future__ import annotations

import unittest

from stream4d_native.v53_reprojection_ledger import (
    _mask_key,
    _passes_candidate_conflict_veto,
    _repeated_support_signature_candidates,
)


class V53ReprojectionLedgerTest(unittest.TestCase):
    def test_mask_key_uses_scene_frame_mask_id(self) -> None:
        self.assertEqual(_mask_key("s", "3", "7"), ("s", 3, 7))

    def test_candidate_conflict_veto_requires_rows_and_threshold(self) -> None:
        self.assertTrue(_passes_candidate_conflict_veto(0.05, 3, 0.05))
        self.assertFalse(_passes_candidate_conflict_veto(0.06, 3, 0.05))
        self.assertFalse(_passes_candidate_conflict_veto(0.0, 0, 0.05))
        self.assertTrue(_passes_candidate_conflict_veto(0.90, 3, None))

    def test_repeated_support_signature_candidates_group_components_without_gt_prediction(self) -> None:
        representative_by_mask = {
            "s:0:1": {"mask_observation_id": "s:0:1", "scene": "s", "chunk_id": "s:chunk000", "frame_id": "0", "mask_id": "1"},
            "s:1:2": {"mask_observation_id": "s:1:2", "scene": "s", "chunk_id": "s:chunk000", "frame_id": "1", "mask_id": "2"},
        }
        support_rows = [
            {"scene": "s", "frame_id": "0", "mask_observation_id": "s:0:1", "mask_id": "1", "component_id": "c1", "support_count": "3", "W_visible": "1.0"},
            {"scene": "s", "frame_id": "1", "mask_observation_id": "s:1:2", "mask_id": "2", "component_id": "c1", "support_count": "4", "W_visible": "0.8"},
            {"scene": "s", "frame_id": "0", "mask_observation_id": "s:0:1", "mask_id": "1", "component_id": "c2", "support_count": "5", "W_visible": "1.0"},
            {"scene": "s", "frame_id": "1", "mask_observation_id": "s:1:2", "mask_id": "2", "component_id": "c2", "support_count": "6", "W_visible": "0.8"},
            {"scene": "s", "frame_id": "0", "mask_observation_id": "s:0:1", "mask_id": "1", "component_id": "c3", "support_count": "1", "W_visible": "1.0"},
        ]
        candidates = _repeated_support_signature_candidates(
            support_rows=support_rows,
            representative_by_mask=representative_by_mask,
            seen_component_sets=set(),
            start_index=7,
            min_frames=2,
            min_components=2,
            min_w_visible=0.5,
            max_components=8,
            max_groups_per_scene=4,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_id"], "cand00007")
        self.assertEqual(candidates[0]["candidate_source"], "R5_repeated_support_signature")
        self.assertEqual(candidates[0]["component_ids"], ["c2", "c1"])
        self.assertFalse(candidates[0]["uses_gt_for_prediction"])


if __name__ == "__main__":
    unittest.main()
